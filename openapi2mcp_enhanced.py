#!/usr/bin/env python3
"""
Comprehensive OpenAPI to MCP Tool Generator
Supports both Swagger 2.0 and OpenAPI 3.x specifications

Features:
- Full Swagger 2.0 and OpenAPI 3.x support
- anyOf, oneOf, allOf schema combinations
- Discriminators for polymorphic types
- API discovery tools (list, schema, invoke)
- Tag-based organization
- Complex nested schemas
- Full $ref resolution with prance
- Multiple authentication schemes
- Request/response examples
- Custom extensions

Usage:
    python openapi2mcp.py --spec api.yaml --output mcp_tools.py
    python openapi2mcp.py --url https://api.example.com/openapi.json --name "My API"
"""

import argparse
import json
import yaml
import logging
import asyncio
import httpx
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
import re
from jinja2 import Template
import sys
from datetime import datetime

# Configure logging first - but don't set level here, let main() handle it
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARNING  # Default to WARNING, will be changed by CLI args
)
logger = logging.getLogger(__name__)

# Try to import prance - provide helpful error if missing
try:
    import prance
    PRANCE_AVAILABLE = True
except ImportError:
    PRANCE_AVAILABLE = False
    logger.warning("prance library not available - $ref resolution will be limited")

@dataclass
class ParameterSpec:
    """Comprehensive parameter specification."""
    name: str
    location: str  # path, query, header, cookie, body
    required: bool
    python_type: str
    openapi_type: str
    description: Optional[str] = None
    default: Any = None
    enum: Optional[List[Any]] = None
    format: Optional[str] = None
    pattern: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    discriminator: Optional[str] = None
    discriminator_mapping: Optional[Dict[str, str]] = None
    possible_types: List[str] = field(default_factory=list)
    collection_format: Optional[str] = None  # Swagger 2.0
    example: Any = None
    examples: Optional[Dict[str, Any]] = None

@dataclass
class ResponseSpec:
    """Response specification."""
    status_code: str
    description: str
    schema: Optional[Dict[str, Any]] = None
    examples: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, Any]] = None

@dataclass
class MCPToolSpec:
    """Comprehensive MCP Tool specification."""
    name: str
    description: str
    method: str
    path: str
    parameters: List[ParameterSpec]
    request_body_params: List[ParameterSpec]
    headers: List[ParameterSpec]
    responses: List[ResponseSpec]
    auth_required: bool = True
    auth_schemes: List[str] = field(default_factory=list)
    rate_limit: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    has_union_types: bool = False
    deprecated: bool = False
    operation_id: Optional[str] = None
    consumes: List[str] = field(default_factory=list)  # Swagger 2.0
    produces: List[str] = field(default_factory=list)  # Swagger 2.0

class ComprehensiveOpenAPIParser:
    """Parse OpenAPI/Swagger specifications comprehensively."""
    
    def __init__(self, spec_path: str, auto_fix_headers: bool = False, 
                 additional_header_patterns: List[str] = None):
        logger.info(f"Initializing parser with spec path: {spec_path}")
        
        self.auto_fix_headers = auto_fix_headers
        self.additional_header_patterns = additional_header_patterns or []
        
        # Load specification
        self.spec = self._load_specification(spec_path)
        
        # Try to resolve references if prance is available
        if PRANCE_AVAILABLE and not spec_path.startswith(('http://', 'https://')):
            try:
                logger.info("Attempting to resolve $refs with prance...")
                resolver = prance.ResolvingParser(spec_path, lazy=False)
                self.spec = resolver.specification
                logger.info("Successfully resolved all $refs")
            except Exception as e:
                logger.warning(f"Prance failed to resolve refs, using unresolved spec: {e}")
        
        self.spec_version = self._detect_spec_version()
        logger.info(f"Detected spec version: {self.spec_version}")
        
        self.base_url = self._extract_base_url()
        self.security_schemes = self._extract_security_schemes()
        self.schema_cache = {}
        
        # Global API info
        self.api_info = self.spec.get('info', {})
        self.api_title = self.api_info.get('title', 'API')
        self.api_version = self.api_info.get('version', '1.0.0')
        self.api_description = self.api_info.get('description', '')
        
        logger.info(f"Parsed {self.spec_version} spec: {self.api_title} v{self.api_version}")
    
    def _load_specification(self, spec_path: str) -> Dict[str, Any]:
        """Load specification from file or URL."""
        if spec_path.startswith(('http://', 'https://')):
            # Load from URL
            logger.info(f"Loading specification from URL: {spec_path}")
            try:
                import requests
                response = requests.get(spec_path, timeout=30)
                response.raise_for_status()
                content = response.text
                
                # Try to parse as JSON first, then YAML
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return yaml.safe_load(content)
            except Exception as e:
                raise ValueError(f"Failed to load specification from URL: {e}")
        else:
            # Load from file
            spec_path_obj = Path(spec_path)
            if not spec_path_obj.exists():
                raise FileNotFoundError(f"Specification file not found: {spec_path}")
            
            logger.info(f"Found spec file at: {spec_path_obj.absolute()}")
            
            with open(spec_path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect format by content if extension is ambiguous
            if spec_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(content)
            elif spec_path.endswith('.json'):
                return json.loads(content)
            else:
                # Try JSON first, then YAML
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    try:
                        return yaml.safe_load(content)
                    except yaml.YAMLError:
                        raise ValueError("Could not parse specification as JSON or YAML")
    
    def _detect_spec_version(self) -> str:
        """Detect specification version."""
        if 'swagger' in self.spec:
            version = self.spec.get('swagger', '2.0')
            return f"swagger_{version.replace('.', '_')}"
        elif 'openapi' in self.spec:
            version = self.spec.get('openapi', '3.0.0')
            return f"openapi_{version.split('.')[0]}"
        return "unknown"
    
    def _extract_base_url(self) -> str:
        """Extract base URL from specification."""
        if self.spec_version.startswith('openapi'):
            # OpenAPI 3.x
            if 'servers' in self.spec and self.spec['servers']:
                server = self.spec['servers'][0]
                url = server['url']
                # Handle variables in URL
                if 'variables' in server:
                    for var_name, var_def in server['variables'].items():
                        default_value = var_def.get('default', '')
                        url = url.replace(f'{{{var_name}}}', str(default_value))
                return url
        else:
            # Swagger 2.0
            schemes = self.spec.get('schemes', ['https'])
            host = self.spec.get('host', 'api.example.com')
            base_path = self.spec.get('basePath', '/')
            return f"{schemes[0]}://{host}{base_path}"
        
        return "https://api.example.com"
    
    def _extract_security_schemes(self) -> Dict[str, Any]:
        """Extract security schemes from specification."""
        if self.spec_version.startswith('openapi'):
            # OpenAPI 3.x
            if 'components' in self.spec and 'securitySchemes' in self.spec['components']:
                return self.spec['components']['securitySchemes']
        else:
            # Swagger 2.0
            if 'securityDefinitions' in self.spec:
                return self.spec['securityDefinitions']
        return {}
    
    def parse_paths(self) -> List[MCPToolSpec]:
        """Parse all paths into MCP tool specifications."""
        tools = []
        paths = self.spec.get('paths', {})
        
        for path, path_spec in paths.items():
            # Handle path-level parameters
            path_params = path_spec.get('parameters', [])
            
            for method, operation in path_spec.items():
                if method.lower() not in ['get', 'post', 'put', 'patch', 'delete', 'head', 'options']:
                    continue
                
                if isinstance(operation, dict):
                    tool = self._create_tool_spec(path, method.upper(), operation, path_params)
                    if tool:
                        tools.append(tool)
        
        logger.info(f"Parsed {len(tools)} operations from {len(paths)} paths")
        return tools
    
    def _create_tool_spec(self, path: str, method: str, operation: Dict[str, Any], 
                         path_params: List[Dict[str, Any]]) -> Optional[MCPToolSpec]:
        """Create comprehensive MCP tool specification."""
        try:
            # Generate tool name
            operation_id = operation.get('operationId')
            if operation_id:
                tool_name = self._sanitize_tool_name(operation_id)
            else:
                tool_name = self._generate_tool_name(path, method)
            
            # Extract metadata
            description = operation.get('summary', operation.get('description', f"{method} {path}"))
            deprecated = operation.get('deprecated', False)
            tags = operation.get('tags', [])
            
            # Merge path-level and operation-level parameters
            all_params = path_params + operation.get('parameters', [])
            
            # Parse parameters
            parameters = []
            headers = []
            request_body_params = []
            has_union_types = False
            
            # For Swagger 2.0
            consumes = operation.get('consumes', self.spec.get('consumes', ['application/json']))
            produces = operation.get('produces', self.spec.get('produces', ['application/json']))
            
            # Validate body parameters for appropriate methods
            if self.spec_version.startswith('swagger'):
                if method.upper() in ['GET', 'DELETE', 'HEAD'] and any(p.get('in') == 'body' for p in all_params):
                    logger.warning(f"{method} {path} has body parameters but typically shouldn't")
            
            # Process parameters
            for param in all_params:
                if self.spec_version.startswith('swagger'):
                    # Swagger 2.0 handling
                    if param.get('in') == 'body':
                        # Swagger 2.0 specific: only one body parameter allowed per operation
                        body_params = self._parse_swagger2_body_parameter(param)
                        request_body_params.extend(body_params)
                        for bp in body_params:
                            if bp.possible_types:
                                has_union_types = True
                    elif param.get('in') == 'formData':
                        # Form data parameters
                        param_spec = self._parse_form_parameter(param)
                        if param_spec:
                            request_body_params.append(param_spec)
                    else:
                        param_spec = self._parse_parameter(param)
                        if param_spec:
                            if param_spec.location == 'header':
                                headers.append(param_spec)
                            else:
                                parameters.append(param_spec)
                            if param_spec.possible_types:
                                has_union_types = True
                else:
                    # OpenAPI 3.x handling
                    param_spec = self._parse_parameter(param)
                    if param_spec:
                        if param_spec.location == 'header':
                            headers.append(param_spec)
                        else:
                            parameters.append(param_spec)
                        if param_spec.possible_types:
                            has_union_types = True
            
            # Parse request body for OpenAPI 3.x
            if self.spec_version.startswith('openapi') and 'requestBody' in operation:
                content = operation['requestBody'].get('content', {})
                if not content:
                    logger.warning(f"{method} {path} has requestBody but no content types")
                body_params = self._parse_request_body(operation['requestBody'])
                request_body_params.extend(body_params)
                for param in body_params:
                    if param.possible_types:
                        has_union_types = True
            
            # Validate path parameters exist
            path_param_names = set(re.findall(r'\{([^}]+)\}', path))
            parsed_path_params = {p.name for p in parameters if p.location == 'path'}
            
            # Check for missing path parameters
            missing_path_params = path_param_names - parsed_path_params
            if missing_path_params:
                logger.warning(f"Missing path parameter definitions for {method} {path}: {missing_path_params}")
                
                # Try to find these parameters in other locations
                for missing_param in missing_path_params:
                    # Check if it's defined but with wrong location
                    found_in_other_location = False
                    
                    # Check in all parsed parameters
                    for param_list in [parameters, request_body_params, headers]:
                        for param in param_list:
                            if param.name == missing_param:
                                logger.info(f"Found '{missing_param}' defined as '{param.location}' parameter, moving to path")
                                # Remove from current location
                                param_list.remove(param)
                                # Update location and add to path parameters
                                param.location = 'path'
                                param.required = True  # Path params are always required
                                parameters.append(param)
                                found_in_other_location = True
                                break
                        if found_in_other_location:
                            break
                    
                    # If still not found, create a default path parameter
                    if not found_in_other_location:
                        logger.info(f"Creating default path parameter for '{missing_param}'")
                        # Try to guess the type based on the parameter name
                        param_type = 'string'
                        if any(suffix in missing_param.lower() for suffix in ['id', 'number', 'code']):
                            # Could be string or integer, default to string for safety
                            param_type = 'string'
                        
                        parameters.append(ParameterSpec(
                            name=missing_param,
                            location='path',
                            required=True,  # Path params are always required
                            python_type='str',
                            openapi_type=param_type,
                            description=f'Path parameter {missing_param}'
                        ))
            
            # Parse responses
            responses = self._parse_responses(operation.get('responses', {}))
            
            # Extract security requirements
            security = operation.get('security', self.spec.get('security', []))
            auth_required = bool(security)
            auth_schemes = []
            if security:
                for sec_req in security:
                    auth_schemes.extend(list(sec_req.keys()))
            
            # Extract rate limiting (custom extension)
            rate_limit = operation.get('x-rate-limit')
            
            return MCPToolSpec(
                name=tool_name,
                description=description,
                method=method,
                path=path,
                parameters=parameters,
                request_body_params=request_body_params,
                headers=headers,
                responses=responses,
                auth_required=auth_required,
                auth_schemes=auth_schemes,
                rate_limit=rate_limit,
                tags=tags,
                has_union_types=has_union_types,
                deprecated=deprecated,
                operation_id=operation_id,
                consumes=consumes,
                produces=produces
            )
            
        except Exception as e:
            logger.error(f"Failed to parse operation {method} {path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_parameter(self, param: Dict[str, Any]) -> Optional[ParameterSpec]:
        """Parse parameter with full support for all specifications."""
        try:
            name = param.get('name')
            if not name:
                return None
            
            location = param.get('in', 'query')
            
            # Special handling for common header patterns
            header_patterns = [
                'authorization', 'auth', 'token', 'api-key', 'x-',
                'content-type', 'accept', 'user-agent', 'cookie',
                'referer', 'origin', 'host', 'connection',
                # Add more banking-specific header patterns
                'channel', 'correlation', 'session', 'trace',
                'request-id', 'client-id', 'tenant-id'
            ] + self.additional_header_patterns
            
            # Check for parameters that end with common header suffixes
            header_suffixes = ['token', 'key', 'id', 'code', 'flag', 'type', 'mode']
            
            # If it looks like a header but isn't marked as one
            if location != 'header':
                name_lower = name.lower()
                is_likely_header = False
                
                # Check if it matches header patterns
                if any(pattern in name_lower for pattern in header_patterns):
                    is_likely_header = True
                # Check if it's a flag/mode/type that might be a header
                elif any(name_lower.endswith(suffix) for suffix in header_suffixes) and location == 'query':
                    # Check if it's likely a control parameter (often headers in banking APIs)
                    control_keywords = ['authorise', 'authorize', 'validate', 'verify', 'mode', 'channel', 'system']
                    if any(keyword in name_lower for keyword in control_keywords):
                        is_likely_header = True
                
                if is_likely_header:
                    if self.auto_fix_headers:
                        logger.info(f"Auto-fixing: Moving parameter '{name}' from '{location}' to 'header'")
                        location = 'header'
                        # Update description to note the auto-fix
                        description = f"{description} (auto-moved to header)" if description else "Auto-moved to header"
                    else:
                        logger.warning(f"Parameter '{name}' appears to be a header but marked as '{location}'")
            
            # Handle required field
            if location == 'path':
                required = True  # Path params always required
            else:
                required = param.get('required', False)
            
            description = param.get('description', '')
            
            # Get schema - handle both OpenAPI 3.x and Swagger 2.0
            if 'schema' in param:
                # OpenAPI 3.x style
                schema = param['schema']
            elif 'content' in param:
                # OpenAPI 3.x with content negotiation
                content = param['content']
                # Look for JSON content first
                for content_type, content_spec in content.items():
                    if 'json' in content_type:
                        schema = content_spec.get('schema', {})
                        break
                else:
                    # Use first available content type
                    schema = next(iter(content.values())).get('schema', {})
            else:
                # Swagger 2.0 style
                schema = self._build_schema_from_parameter(param)
            
            # Handle complex schemas (anyOf, oneOf, allOf)
            if any(key in schema for key in ['anyOf', 'oneOf', 'allOf']):
                return self._parse_union_parameter(name, location, required, description, schema, param)
            
            # Handle schema with multiple types
            if 'type' in schema and isinstance(schema['type'], list):
                # Multiple types specified
                return self._parse_multi_type_parameter(name, location, required, description, schema, param)
            
            # Extract examples
            example = param.get('example', schema.get('example'))
            examples = param.get('examples', schema.get('examples'))
            
            # Determine types
            python_type = self._get_python_type(schema)
            openapi_type = schema.get('type', 'string')
            
            return ParameterSpec(
                name=name,
                location=location,
                required=required,
                python_type=python_type,
                openapi_type=openapi_type,
                description=description,
                default=schema.get('default', param.get('default')),
                enum=schema.get('enum', param.get('enum')),
                format=schema.get('format', param.get('format')),
                pattern=schema.get('pattern', param.get('pattern')),
                minimum=schema.get('minimum', param.get('minimum')),
                maximum=schema.get('maximum', param.get('maximum')),
                min_length=schema.get('minLength', param.get('minLength')),
                max_length=schema.get('maxLength', param.get('maxLength')),
                collection_format=param.get('collectionFormat'),
                example=example,
                examples=examples
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse parameter {param.get('name', 'unknown')}: {e}")
            return None
    
    def _build_schema_from_parameter(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """Build schema object from Swagger 2.0 parameter."""
        schema = {
            'type': param.get('type', 'string'),
            'format': param.get('format'),
            'enum': param.get('enum'),
            'default': param.get('default'),
            'minimum': param.get('minimum'),
            'maximum': param.get('maximum'),
            'exclusiveMinimum': param.get('exclusiveMinimum'),
            'exclusiveMaximum': param.get('exclusiveMaximum'),
            'minLength': param.get('minLength'),
            'maxLength': param.get('maxLength'),
            'pattern': param.get('pattern'),
            'minItems': param.get('minItems'),
            'maxItems': param.get('maxItems'),
            'uniqueItems': param.get('uniqueItems'),
            'multipleOf': param.get('multipleOf')
        }
        
        # Handle array items
        if param.get('type') == 'array' and 'items' in param:
            schema['items'] = param['items']
        
        # Handle collection format for arrays
        if param.get('collectionFormat'):
            schema['x-collection-format'] = param['collectionFormat']
        
        # Remove None values
        return {k: v for k, v in schema.items() if v is not None}
    
    def _parse_form_parameter(self, param: Dict[str, Any]) -> Optional[ParameterSpec]:
        """Parse form data parameter (Swagger 2.0)."""
        # Form parameters are similar to regular parameters but in the body
        param_spec = self._parse_parameter(param)
        if param_spec:
            param_spec.location = 'body'
        return param_spec
    
    def _parse_multi_type_parameter(self, name: str, location: str, required: bool,
                                   description: str, schema: Dict[str, Any],
                                   original_param: Dict[str, Any]) -> ParameterSpec:
        """Parse parameter with multiple types specified."""
        types = schema.get('type', [])
        if not isinstance(types, list):
            types = [types]
        
        # Convert OpenAPI types to Python types
        possible_types = []
        for t in types:
            temp_schema = {'type': t, 'format': schema.get('format')}
            python_type = self._get_base_python_type(temp_schema)
            if python_type not in possible_types:
                possible_types.append(python_type)
        
        # Create Union type
        if len(possible_types) == 0:
            python_type = 'Any'
        elif len(possible_types) == 1:
            python_type = possible_types[0]
        else:
            unique_types = list(dict.fromkeys(possible_types))
            python_type = f'Union[{", ".join(unique_types)}]'
        
        # Extract example
        example = original_param.get('example', schema.get('example'))
        examples = original_param.get('examples', schema.get('examples'))
        
        return ParameterSpec(
            name=name,
            location=location,
            required=required,
            python_type=python_type,
            openapi_type='multiple',
            description=description,
            possible_types=possible_types,
            default=schema.get('default'),
            enum=schema.get('enum'),
            format=schema.get('format'),
            pattern=schema.get('pattern'),
            minimum=schema.get('minimum'),
            maximum=schema.get('maximum'),
            min_length=schema.get('minLength'),
            max_length=schema.get('maxLength'),
            example=example,
            examples=examples
        )
    
    def _parse_union_parameter(self, name: str, location: str, required: bool,
                              description: str, schema: Dict[str, Any],
                              original_param: Dict[str, Any]) -> ParameterSpec:
        """Parse parameter with anyOf, oneOf, or allOf."""
        possible_types = []
        discriminator = None
        discriminator_mapping = None
        
        # Extract discriminator
        if 'discriminator' in schema:
            if isinstance(schema['discriminator'], dict):
                # OpenAPI 3.x style
                discriminator = schema['discriminator'].get('propertyName')
                discriminator_mapping = schema['discriminator'].get('mapping', {})
                
                # Add discriminator info to description
                if discriminator_mapping:
                    description += f"\nDiscriminator: {discriminator} with mapping: {discriminator_mapping}"
            else:
                # Swagger 2.0 style
                discriminator = schema['discriminator']
        
        # Process anyOf
        if 'anyOf' in schema:
            for sub_schema in schema['anyOf']:
                type_str = self._get_python_type(sub_schema)
                if type_str not in possible_types:
                    possible_types.append(type_str)
                    
                # If it's a path parameter with anyOf, add validation info
                if location == 'path':
                    if 'pattern' in sub_schema:
                        description += f"\nPattern for {sub_schema.get('type', 'type')}: {sub_schema['pattern']}"
                    if 'format' in sub_schema:
                        description += f"\nFormat: {sub_schema['format']}"
        
        # Process oneOf
        elif 'oneOf' in schema:
            for i, sub_schema in enumerate(schema['oneOf']):
                type_str = self._get_python_type(sub_schema)
                if type_str not in possible_types:
                    possible_types.append(type_str)
                    
                # Add specific validation rules for each option
                if 'pattern' in sub_schema:
                    description += f"\nOption {i+1} pattern: {sub_schema['pattern']}"
                if 'enum' in sub_schema:
                    description += f"\nOption {i+1} values: {sub_schema['enum']}"
                if 'format' in sub_schema:
                    description += f"\nOption {i+1} format: {sub_schema['format']}"
        
        # Process allOf
        elif 'allOf' in schema:
            # Merge all schemas
            merged_schema = self._merge_all_of_schemas(schema['allOf'])
            possible_types = [self._get_python_type(merged_schema)]
        
        # Create Union type
        if len(possible_types) == 0:
            python_type = 'Any'
        elif len(possible_types) == 1:
            python_type = possible_types[0]
        else:
            unique_types = list(dict.fromkeys(possible_types))
            python_type = f'Union[{", ".join(unique_types)}]'
        
        # Determine OpenAPI type
        openapi_type = 'union'
        if 'anyOf' in schema:
            openapi_type = 'anyOf'
        elif 'oneOf' in schema:
            openapi_type = 'oneOf'
        elif 'allOf' in schema:
            openapi_type = 'allOf'
        
        # Extract example
        example = original_param.get('example', schema.get('example'))
        examples = original_param.get('examples', schema.get('examples'))
        
        # For path parameters with unions, provide helpful examples
        if location == 'path' and not example and not examples:
            # Generate examples based on the possible types
            generated_examples = []
            if 'anyOf' in schema:
                for sub_schema in schema['anyOf']:
                    if 'example' in sub_schema:
                        generated_examples.append(sub_schema['example'])
                    elif 'enum' in sub_schema and sub_schema['enum']:
                        generated_examples.append(sub_schema['enum'][0])
            elif 'oneOf' in schema:
                for sub_schema in schema['oneOf']:
                    if 'example' in sub_schema:
                        generated_examples.append(sub_schema['example'])
                    elif 'enum' in sub_schema and sub_schema['enum']:
                        generated_examples.append(sub_schema['enum'][0])
            
            if generated_examples:
                examples = {"generated": generated_examples}
                description += f"\nExamples: {', '.join(str(ex) for ex in generated_examples)}"
        
        return ParameterSpec(
            name=name,
            location=location,
            required=required,
            python_type=python_type,
            openapi_type=openapi_type,
            description=description,
            discriminator=discriminator,
            discriminator_mapping=discriminator_mapping,
            possible_types=possible_types,
            example=example,
            examples=examples
        )
    
    def _merge_all_of_schemas(self, schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple schemas for allOf."""
        merged = {
            'type': 'object',
            'properties': {},
            'required': []
        }
        
        for schema in schemas:
            if 'type' in schema and merged.get('type') == 'object':
                merged['type'] = schema['type']
            
            if 'properties' in schema:
                merged['properties'].update(schema['properties'])
            
            if 'required' in schema:
                merged['required'].extend(schema['required'])
            
            # Copy other properties
            for key, value in schema.items():
                if key not in ['type', 'properties', 'required']:
                    merged[key] = value
        
        # Remove duplicates from required
        merged['required'] = list(dict.fromkeys(merged['required']))
        
        return merged
    
    def _parse_swagger2_body_parameter(self, param: Dict[str, Any]) -> List[ParameterSpec]:
        """Parse Swagger 2.0 body parameter."""
        params = []
        
        try:
            if param.get('in') != 'body':
                return params
            
            name = param.get('name', 'body')
            required = param.get('required', True)
            description = param.get('description', '')
            
            # Get the schema
            schema = param.get('schema', {})
            
            # Check for file uploads (Swagger 2.0 specific)
            if schema.get('type') == 'file':
                return [ParameterSpec(
                    name=name,
                    location='body',
                    required=required,
                    python_type='bytes',
                    openapi_type='file',
                    description=description or 'File upload',
                    format='binary',
                    example=param.get('example', schema.get('example'))
                )]
            
            # Handle union types at root
            if any(key in schema for key in ['anyOf', 'oneOf', 'allOf']):
                param_spec = self._parse_union_parameter(
                    'request_body',
                    'body',
                    required,
                    description,
                    schema,
                    param
                )
                params.append(param_spec)
                return params
            
            # If not an object or no properties, return single parameter
            if schema.get('type') != 'object' or 'properties' not in schema:
                param_spec = ParameterSpec(
                    name='request_body',
                    location='body',
                    required=required,
                    python_type=self._get_python_type(schema),
                    openapi_type=schema.get('type', 'any'),
                    description=description,
                    example=param.get('example', schema.get('example'))
                )
                params.append(param_spec)
                return params
            
            # Expand object properties
            properties = schema.get('properties', {})
            required_props = schema.get('required', [])
            
            for prop_name, prop_schema in properties.items():
                param_spec = self._parse_schema_property(
                    prop_name,
                    prop_schema,
                    prop_name in required_props
                )
                if param_spec:
                    params.append(param_spec)
            
            return params
            
        except Exception as e:
            logger.warning(f"Failed to parse body parameter: {e}")
            return params
    
    def _parse_request_body(self, request_body: Dict[str, Any]) -> List[ParameterSpec]:
        """Parse OpenAPI 3.x request body."""
        params = []
        
        try:
            content = request_body.get('content', {})
            required = request_body.get('required', True)
            description = request_body.get('description', '')
            
            # Look for JSON content first, then any content type
            json_content = None
            for content_type, content_spec in content.items():
                if 'json' in content_type:
                    json_content = content_spec
                    break
            
            if not json_content and content:
                # Use first available content type
                json_content = next(iter(content.values()))
            
            if not json_content:
                return params
            
            schema = json_content.get('schema', {})
            examples = json_content.get('examples', {})
            example = json_content.get('example')
            
            # Handle union types at root
            if any(key in schema for key in ['anyOf', 'oneOf', 'allOf']):
                param = self._parse_union_parameter(
                    'request_body',
                    'body',
                    required,
                    description,
                    schema,
                    {'example': example, 'examples': examples}
                )
                params.append(param)
                return params
            
            # Handle object schemas
            if schema.get('type') == 'object' and 'properties' in schema:
                properties = schema.get('properties', {})
                required_props = schema.get('required', [])
                
                for prop_name, prop_schema in properties.items():
                    param = self._parse_schema_property(
                        prop_name,
                        prop_schema,
                        prop_name in required_props
                    )
                    if param:
                        params.append(param)
            else:
                # Single non-object body
                param = ParameterSpec(
                    name='request_body',
                    location='body',
                    required=required,
                    python_type=self._get_python_type(schema),
                    openapi_type=schema.get('type', 'any'),
                    description=description,
                    example=example,
                    examples=examples
                )
                params.append(param)
            
            return params
            
        except Exception as e:
            logger.warning(f"Failed to parse request body: {e}")
            return params
    
    def _parse_schema_property(self, name: str, schema: Dict[str, Any], 
                               required: bool) -> Optional[ParameterSpec]:
        """Parse a schema property into ParameterSpec."""
        try:
            # Handle union types
            if any(key in schema for key in ['anyOf', 'oneOf', 'allOf']):
                return self._parse_union_parameter(
                    name,
                    'body',
                    required,
                    schema.get('description', ''),
                    schema,
                    schema
                )
            
            # Regular property
            return ParameterSpec(
                name=name,
                location='body',
                required=required,
                python_type=self._get_python_type(schema),
                openapi_type=schema.get('type', 'string'),
                description=schema.get('description', ''),
                default=schema.get('default'),
                enum=schema.get('enum'),
                format=schema.get('format'),
                pattern=schema.get('pattern'),
                minimum=schema.get('minimum'),
                maximum=schema.get('maximum'),
                min_length=schema.get('minLength'),
                max_length=schema.get('maxLength'),
                example=schema.get('example'),
                examples=schema.get('examples')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse property {name}: {e}")
            return None
    
    def _parse_responses(self, responses: Dict[str, Any]) -> List[ResponseSpec]:
        """Parse operation responses."""
        response_specs = []
        
        for status_code, response in responses.items():
            if isinstance(response, dict):
                response_spec = ResponseSpec(
                    status_code=str(status_code),
                    description=response.get('description', ''),
                    headers=response.get('headers'),
                    examples=response.get('examples')
                )
                
                # Extract schema
                if self.spec_version.startswith('swagger'):
                    # Swagger 2.0
                    response_spec.schema = response.get('schema')
                else:
                    # OpenAPI 3.x
                    content = response.get('content', {})
                    for content_type, content_spec in content.items():
                        if 'json' in content_type:
                            response_spec.schema = content_spec.get('schema')
                            break
                
                response_specs.append(response_spec)
        
        return response_specs
    
    def _get_python_type(self, schema: Dict[str, Any]) -> str:
        """Convert OpenAPI schema to Python type annotation."""
        # Handle nullable
        nullable = schema.get('nullable', False)
        if self.spec_version.startswith('swagger'):
            # In Swagger 2.0, check for x-nullable extension
            nullable = schema.get('x-nullable', False)
        
        base_type = self._get_base_python_type(schema)
        
        if nullable and not base_type.startswith('Optional['):
            return f'Optional[{base_type}]'
        
        return base_type
    
    def _get_base_python_type(self, schema: Dict[str, Any]) -> str:
        """Get base Python type without Optional wrapper."""
        openapi_type = schema.get('type', 'string')
        format_type = schema.get('format', '')
        
        # Handle arrays
        if openapi_type == 'array':
            items = schema.get('items', {})
            items_type = self._get_python_type(items)
            return f'List[{items_type}]'
        
        # Handle objects
        if openapi_type == 'object':
            if 'properties' in schema:
                # Could generate TypedDict here
                additional_props = schema.get('additionalProperties')
                if isinstance(additional_props, dict):
                    value_type = self._get_python_type(additional_props)
                    return f'Dict[str, {value_type}]'
                return 'Dict[str, Any]'
            elif 'additionalProperties' in schema:
                additional_props = schema['additionalProperties']
                if isinstance(additional_props, dict):
                    value_type = self._get_python_type(additional_props)
                    return f'Dict[str, {value_type}]'
                elif additional_props is True:
                    return 'Dict[str, Any]'
                else:
                    return 'Dict[str, Any]'
            return 'Dict[str, Any]'
        
        # Handle references that weren't resolved
        if '$ref' in schema:
            ref_name = schema['$ref'].split('/')[-1]
            return f'Dict[str, Any]  # {ref_name}'
        
        # Handle enums
        if 'enum' in schema and schema['enum']:
            # For string enums, use Literal
            if all(isinstance(v, str) for v in schema['enum']):
                enum_values = ', '.join(f'"{v}"' for v in schema['enum'])
                return f'Literal[{enum_values}]'
            else:
                # Mixed types
                return 'Union[str, int, float]'
        
        # Basic type mapping
        type_map = {
            ('string', ''): 'str',
            ('string', 'date'): 'str',  # Could use datetime.date
            ('string', 'date-time'): 'str',  # Could use datetime
            ('string', 'password'): 'str',
            ('string', 'email'): 'str',
            ('string', 'uri'): 'str',
            ('string', 'url'): 'str',
            ('string', 'uuid'): 'str',
            ('string', 'binary'): 'bytes',
            ('string', 'byte'): 'str',  # Base64
            ('integer', ''): 'int',
            ('integer', 'int32'): 'int',
            ('integer', 'int64'): 'int',
            ('number', ''): 'float',
            ('number', 'float'): 'float',
            ('number', 'double'): 'float',
            ('number', 'decimal'): 'Decimal',
            ('boolean', ''): 'bool',
            ('file', ''): 'bytes'  # Swagger 2.0 file upload
        }
        
        python_type = type_map.get((openapi_type, format_type), 'Any')
        
        # Handle type arrays (used in some specs)
        if isinstance(openapi_type, list):
            types = [self._get_base_python_type({'type': t}) for t in openapi_type]
            if len(types) == 1:
                return types[0]
            else:
                return f'Union[{", ".join(types)}]'
        
        return python_type
    
    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize operation ID for use as Python function name."""
        # Convert camelCase to snake_case
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        # Replace special characters
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Convert to lowercase
        name = name.lower()
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = f'op_{name}'
        
        # Ensure it's not a Python keyword
        import keyword
        if keyword.iskeyword(name):
            name = f'{name}_op'
        
        return name or 'unnamed_operation'
    
    def _generate_tool_name(self, path: str, method: str) -> str:
        """Generate tool name from path and method."""
        # Remove path parameters
        clean_path = re.sub(r'\{[^}]+\}', '', path)
        
        # Split path into parts
        parts = [p for p in clean_path.split('/') if p]
        
        # Take last 2-3 meaningful parts
        if len(parts) >= 2:
            base_parts = parts[-2:]
        elif parts:
            base_parts = parts
        else:
            base_parts = ['resource']
        
        # Method prefix
        method_prefix = {
            'GET': 'get',
            'POST': 'create',
            'PUT': 'update',
            'PATCH': 'patch',
            'DELETE': 'delete',
            'HEAD': 'check',
            'OPTIONS': 'options'
        }.get(method, method.lower())
        
        # Combine
        name_parts = [method_prefix] + base_parts
        tool_name = '_'.join(name_parts)
        
        return self._sanitize_tool_name(tool_name)


class ComprehensiveMCPToolGenerator:
    """Generate comprehensive FastMCP tool code."""
    
    def __init__(self, parser: ComprehensiveOpenAPIParser):
        self.parser = parser
        self.base_url = parser.base_url
        self.security_schemes = parser.security_schemes
        self.api_title = parser.api_title
        self.api_version = parser.api_version
        self.api_description = parser.api_description
    
    def generate_tools_file(self, tools: List[MCPToolSpec], output_file: str,
                           server_name: Optional[str] = None, include_discovery: bool = True,
                           discovery_only: bool = False):
        """Generate Python file with FastMCP tools."""
        template = Template(self._get_comprehensive_template())
        
        # Determine imports
        imports = self._determine_imports(tools)
        
        # Add imports for discovery tools
        if include_discovery:
            imports['collections'] = ['defaultdict']
        
        # Check features
        has_union_types = any(tool.has_union_types for tool in tools)
        has_auth = any(tool.auth_required for tool in tools)
        has_deprecated = any(tool.deprecated for tool in tools)
        
        # Group tools by tag
        tools_by_tag = {}
        for tool in tools:
            for tag in tool.tags or ['untagged']:
                if tag not in tools_by_tag:
                    tools_by_tag[tag] = []
                tools_by_tag[tag].append(tool)
        
        # Generate content
        content = template.render(
            tools=tools,
            base_url=self.base_url,
            security_schemes=self.security_schemes,
            api_title=self.api_title,
            api_version=self.api_version,
            api_description=self.api_description,
            server_name=server_name or self.api_title,
            has_auth=has_auth,
            has_union_types=has_union_types,
            has_deprecated=has_deprecated,
            imports=imports,
            tools_by_tag=tools_by_tag,
            generated_at=datetime.now().isoformat(),
            spec_version=self.parser.spec_version,
            include_discovery=include_discovery,
            discovery_only=discovery_only
        )
        
        # Write file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if discovery_only:
            logger.info(f"Generated 3 discovery tools (catalog contains {len(tools)} endpoints) in {output_file}")
        elif include_discovery:
            logger.info(f"Generated 3 discovery tools + {len(tools)} typed tools in {output_file}")
        else:
            logger.info(f"Generated {len(tools)} tools in {output_file}")
    
    def _determine_imports(self, tools: List[MCPToolSpec]) -> Dict[str, List[str]]:
        """Determine required imports based on tools."""
        imports = {
            'typing': set(['Dict', 'Any', 'Optional']),
            'decimal': set(),
            'datetime': set(),
            'enum': set()
        }
        
        for tool in tools:
            all_params = tool.parameters + tool.request_body_params + tool.headers
            
            for param in all_params:
                # Parse type annotations
                if 'Union[' in param.python_type:
                    imports['typing'].add('Union')
                if 'List[' in param.python_type:
                    imports['typing'].add('List')
                if 'Literal[' in param.python_type:
                    imports['typing'].add('Literal')
                if 'Tuple[' in param.python_type:
                    imports['typing'].add('Tuple')
                if 'Set[' in param.python_type:
                    imports['typing'].add('Set')
                if 'Decimal' in param.python_type:
                    imports['decimal'].add('Decimal')
                if param.format in ['date', 'date-time']:
                    imports['datetime'].add('datetime')
                if param.discriminator:
                    imports['typing'].add('TypeVar')
                    imports['typing'].add('cast')
        
        # Convert sets to sorted lists and filter empty
        return {k: sorted(list(v)) for k, v in imports.items() if v}
    
    def _get_comprehensive_template(self) -> str:
        """Get comprehensive Jinja2 template."""
        return '''"""
{{ api_title }} - MCP Tools
Generated from {{ spec_version }} specification
Version: {{ api_version }}

{{ api_description }}

Generated at: {{ generated_at }}
"""

import os
import httpx
import logging
from fastmcp import FastMCP
{% if imports.typing -%}
from typing import {{ imports.typing | join(', ') }}
{%- endif %}
{% if imports.decimal -%}
from decimal import {{ imports.decimal | join(', ') }}
{%- endif %}
{% if imports.datetime -%}
from datetime import {{ imports.datetime | join(', ') }}
{%- endif %}
{% if imports.enum -%}
from enum import {{ imports.enum | join(', ') }}
{%- endif %}
{% if include_discovery and imports.collections -%}
from collections import {{ imports.collections | join(', ') }}
{%- endif %}
import asyncio
import json
import re

logger = logging.getLogger(__name__)

# API Configuration
BASE_URL = os.getenv("API_BASE_URL", "{{ base_url }}")
API_KEY = os.getenv("API_KEY")
TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

{% if include_discovery %}
# API Catalog - Generated from OpenAPI spec
API_CATALOG = {
    {% for tool in tools %}
    "{{ tool.name }}": {
        "method": "{{ tool.method }}",
        "path": "{{ tool.path }}",
        "description": {{ tool.description | tojson }},
        "tags": {{ tool.tags | tojson }},
        "deprecated": {{ tool.deprecated | tojson }},
        "auth_required": {{ tool.auth_required | tojson }},
        {% if tool.operation_id %}"operation_id": {{ tool.operation_id | tojson }},{% endif %}
        "parameters": [
            {% for param in tool.parameters + tool.request_body_params %}
            {
                "name": "{{ param.name }}",
                "type": "{{ param.openapi_type }}",
                "location": "{{ param.location }}",
                "required": {{ param.required | tojson }},
                "description": {{ (param.description or '') | tojson }}{% if param.example is not none %},
                "example": {{ param.example | tojson }}{% endif %}{% if param.enum %},
                "enum": {{ param.enum | tojson }}{% endif %}{% if param.pattern %},
                "pattern": {{ param.pattern | tojson }}{% endif %}{% if param.format %},
                "format": "{{ param.format }}"{% endif %}{% if param.possible_types %},
                "possible_types": {{ param.possible_types | tojson }}{% endif %}{% if param.discriminator %},
                "discriminator": "{{ param.discriminator }}"{% endif %}
            }{% if not loop.last %},{% endif %}
            {% endfor %}
        ]
    }{% if not loop.last %},{% endif %}
    {% endfor %}
}

# Build tag-based index
TAG_INDEX = defaultdict(list)
for endpoint_name, endpoint_info in API_CATALOG.items():
    for tag in endpoint_info.get("tags", ["untagged"]):
        TAG_INDEX[tag.lower()].append(endpoint_name)

# Build operation ID index
OPERATION_ID_INDEX = {}
for endpoint_name, endpoint_info in API_CATALOG.items():
    if "operation_id" in endpoint_info:
        OPERATION_ID_INDEX[endpoint_info["operation_id"]] = endpoint_name

{% endif %}

{% if has_auth %}
# Default headers (customize based on your API requirements)
DEFAULT_HEADERS = {
    # Add any default headers your API requires
    # "X-Custom-Header": os.getenv("API_CUSTOM_HEADER", "default-value")
}


def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for API requests."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **DEFAULT_HEADERS
    }
    
    if API_KEY:
        # Determine auth scheme from security definitions
        {% if security_schemes %}
        # Using security scheme from specification
        {% for scheme_name, scheme in security_schemes.items() %}
        {% if loop.first %}
        {% if scheme.type == "apiKey" and scheme.in == "header" %}
        headers["{{ scheme.name }}"] = API_KEY
        {% elif scheme.type == "http" and scheme.scheme == "bearer" %}
        headers["Authorization"] = f"Bearer {API_KEY}"
        {% elif scheme.type == "http" and scheme.scheme == "basic" %}
        headers["Authorization"] = f"Basic {API_KEY}"
        {% endif %}
        {% endif %}
        {% endfor %}
        {% else %}
        # Default to Bearer token
        headers["Authorization"] = f"Bearer {API_KEY}"
        {% endif %}
    
    return headers
{% endif %}


async def make_api_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Make authenticated request to API."""
    url = f"{BASE_URL.rstrip('/')}{path}"
    
    {% if has_auth %}
    request_headers = get_auth_headers()
    {% else %}
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    {% endif %}
    
    if headers:
        request_headers.update(headers)
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            logger.debug(f"API {method} {url}")
            logger.debug(f"Headers: {request_headers}")
            logger.debug(f"Params: {params}")
            logger.debug(f"Data: {data}")
            
            response = await client.request(
                method=method.upper(),
                url=url,
                params=params,
                json=data if data else None,
                headers=request_headers
            )
            
            response.raise_for_status()
            
            # Handle different response types
            content_type = response.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                return response.json()
            else:
                return {"data": response.text, "content_type": content_type}
                
        except httpx.HTTPStatusError as e:
            logger.error(f"API error {e.response.status_code}: {e.response.text}")
            error_detail = {
                "error": True,
                "status_code": e.response.status_code,
                "method": method,
                "url": url,
                "message": "Unknown error"
            }
            
            try:
                error_data = e.response.json()
                error_detail["error_details"] = error_data
                error_detail["message"] = error_data.get("message", error_data.get("error", str(error_data)))
            except:
                error_detail["message"] = e.response.text or str(e)
            
            # Add request details for debugging
            if logger.isEnabledFor(logging.DEBUG):
                error_detail["request"] = {
                    "headers": dict(request_headers),
                    "params": params,
                    "data": data
                }
            
            return error_detail
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            return {
                "error": True,
                "message": str(e)
            }


def sanitize_param_name(name: str) -> str:
    """Convert parameter name to Python-friendly format."""
    # Replace spaces and hyphens with underscores
    name = name.replace(' ', '_').replace('-', '_')
    # Convert to lowercase
    name = name.lower()
    # Remove any other special characters
    name = re.sub(r'[^a-z0-9_]', '', name)
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = f'param_{name}'
    return name

{% if has_union_types %}
def validate_union_type(value: Any, param_name: str, possible_types: List[str]) -> bool:
    """Validate that a value matches one of the possible union types."""
    if value is None:
        return 'None' in possible_types or 'Optional' in str(possible_types)
    
    # Check basic types
    type_checks = {
        'str': lambda v: isinstance(v, str),
        'int': lambda v: isinstance(v, int) and not isinstance(v, bool),
        'float': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        'bool': lambda v: isinstance(v, bool),
        'List': lambda v: isinstance(v, list),
        'Dict': lambda v: isinstance(v, dict),
    }
    
    for possible_type in possible_types:
        # Handle basic types
        for type_name, check_func in type_checks.items():
            if type_name in possible_type and check_func(value):
                return True
        
        # Handle Any type
        if 'Any' in possible_type:
            return True
    
    return False

{% endif %}


def register_{{ server_name | lower | replace(' ', '_') | replace('-', '_') }}_tools(mcp: FastMCP):
    """Register all API tools with FastMCP server."""
    
    {% if include_discovery %}
    # Register API discovery tools
    @mcp.tool()
    async def list_api_endpoints(
        search_query: Optional[str] = None,
        tag: Optional[str] = None,
        method: Optional[str] = None,
        include_deprecated: bool = False
    ) -> Dict[str, Any]:
        """Search and list available API endpoints.
        
        Discover available operations by searching through:
        - search_query: Text to search in names, descriptions, or paths
        - tag: Filter by API domain/tag (e.g., "counterparties", "loans", "accounts")
        - method: Filter by HTTP method (GET, POST, PUT, DELETE, etc.)
        - include_deprecated: Include deprecated endpoints
        
        Returns grouped endpoints with descriptions and metadata.
        """
        results = []
        
        # Filter by tag
        if tag:
            endpoint_names = TAG_INDEX.get(tag.lower(), [])
        else:
            endpoint_names = list(API_CATALOG.keys())
        
        # Apply filters
        for name in endpoint_names:
            endpoint = API_CATALOG[name]
            
            # Skip deprecated if not requested
            if endpoint.get("deprecated") and not include_deprecated:
                continue
            
            # Filter by method
            if method and endpoint["method"] != method.upper():
                continue
            
            # Search filter
            if search_query:
                search_lower = search_query.lower()
                if not any(search_lower in text.lower() for text in [
                    name,
                    endpoint["description"],
                    endpoint["path"],
                    ' '.join(endpoint["tags"])
                ]):
                    continue
            
            results.append({
                "name": name,
                "method": endpoint["method"],
                "path": endpoint["path"],
                "description": endpoint["description"],
                "tags": endpoint["tags"],
                "deprecated": endpoint.get("deprecated", False),
                "auth_required": endpoint.get("auth_required", True),
                "parameter_count": len(endpoint["parameters"])
            })
        
        # Sort results
        results.sort(key=lambda x: (x["deprecated"], x["name"]))
        
        # Group by tags if no specific tag was requested
        if not tag and results:
            by_tag = defaultdict(list)
            for result in results:
                for t in result["tags"] or ["untagged"]:
                    by_tag[t].append(result)
            
            return {
                "total_endpoints": len(results),
                "endpoints_by_tag": dict(by_tag),
                "available_tags": sorted(list(TAG_INDEX.keys()))
            }
        
        return {
            "total_endpoints": len(results),
            "endpoints": results,
            "search_criteria": {
                "query": search_query,
                "tag": tag,
                "method": method,
                "include_deprecated": include_deprecated
            }
        }
    
    @mcp.tool()
    async def get_api_endpoint_schema(
        endpoint_name: Optional[str] = None,
        operation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed schema for a specific API endpoint.
        
        Retrieve by either:
        - endpoint_name: The tool name (from list_api_endpoints)
        - operation_id: The original OpenAPI operationId
        
        Returns complete parameter schemas, types, constraints, and examples.
        """
        # Resolve endpoint name
        if operation_id and operation_id in OPERATION_ID_INDEX:
            endpoint_name = OPERATION_ID_INDEX[operation_id]
        
        if not endpoint_name or endpoint_name not in API_CATALOG:
            suggestions = []
            if endpoint_name:
                # Find similar names
                search_lower = endpoint_name.lower()
                suggestions = [
                    name for name in API_CATALOG.keys()
                    if search_lower in name.lower()
                ][:5]
            
            return {
                "error": True,
                "message": f"Endpoint '{endpoint_name or operation_id}' not found",
                "suggestions": suggestions or list(API_CATALOG.keys())[:10]
            }
        
        endpoint = API_CATALOG[endpoint_name]
        
        # Build parameter details with examples
        parameter_details = []
        for param in endpoint["parameters"]:
            param_detail = {
                "name": param["name"],
                "type": param["type"],
                "location": param["location"],
                "required": param["required"],
                "description": param.get("description", "")
            }
            if "example" in param:
                param_detail["example"] = param["example"]
            parameter_details.append(param_detail)
        
        # Build example call
        example_params = {}
        for param in endpoint["parameters"]:
            if "example" in param:
                example_params[param["name"]] = param["example"]
            elif param["required"]:
                # Generate a placeholder for required params without examples
                example_params[param["name"]] = f"<{param['type']}>"
        
        return {
            "endpoint_name": endpoint_name,
            "operation_id": endpoint.get("operation_id"),
            "method": endpoint["method"],
            "path": endpoint["path"],
            "description": endpoint["description"],
            "tags": endpoint["tags"],
            "deprecated": endpoint.get("deprecated", False),
            "auth_required": endpoint.get("auth_required", True),
            "parameters": parameter_details,
            "example_usage": {
                "endpoint": endpoint_name,
                "params": example_params
            }
        }
    
    @mcp.tool()
    async def invoke_api_endpoint(
        endpoint_name: Optional[str] = None,
        operation_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Dynamically invoke any API endpoint.
        
        Call endpoints discovered through list_api_endpoints:
        - endpoint_name: The tool name
        - operation_id: The original OpenAPI operationId
        - params: Parameters to pass (use get_api_endpoint_schema for requirements)
        
        Validates parameters and executes the API call.
        """
        # Resolve endpoint name
        if operation_id and operation_id in OPERATION_ID_INDEX:
            endpoint_name = OPERATION_ID_INDEX[operation_id]
        
        if not endpoint_name or endpoint_name not in API_CATALOG:
            return {
                "error": True,
                "message": f"Endpoint '{endpoint_name or operation_id}' not found",
                "hint": "Use list_api_endpoints to discover available endpoints"
            }
        
        endpoint = API_CATALOG[endpoint_name]
        params = params or {}
        
        # Validate required parameters
        missing_required = []
        for param_spec in endpoint["parameters"]:
            if param_spec["required"] and param_spec["name"] not in params:
                missing_required.append(param_spec["name"])
        
        if missing_required:
            return {
                "error": True,
                "message": "Missing required parameters",
                "missing_parameters": missing_required,
                "endpoint_schema": {
                    "method": endpoint["method"],
                    "path": endpoint["path"],
                    "parameters": endpoint["parameters"]
                }
            }
        
        # Separate parameters by location
        path_params = {}
        query_params = {}
        body_params = {}
        header_params = {}
        
        for param_spec in endpoint["parameters"]:
            param_name = param_spec["name"]
            if param_name in params:
                value = params[param_name]
                
                # Validate union types if present
                if "possible_types" in param_spec and value is not None:
                    # For path parameters with union types, ensure proper formatting
                    if param_spec["location"] == "path":
                        # Convert to string for path parameters
                        value = str(value)
                        
                        # Apply pattern validation if specified
                        if "pattern" in param_spec:
                            import re
                            if not re.match(param_spec["pattern"], value):
                                return {
                                    "error": True,
                                    "message": f"Path parameter '{param_name}' does not match required pattern",
                                    "pattern": param_spec["pattern"],
                                    "value": value
                                }
                
                if param_spec["location"] == "path":
                    path_params[param_name] = value
                elif param_spec["location"] == "query":
                    query_params[param_name] = value
                elif param_spec["location"] == "body":
                    body_params[param_name] = value
                elif param_spec["location"] == "header":
                    header_params[param_name] = value
        
        # Build path with parameters
        path = endpoint["path"]
        for param_name, param_value in path_params.items():
            path = path.replace(f"{{ '{' }}{param_name}{{ '}' }}", str(param_value))
        
        # Make request
        return await make_api_request(
            method=endpoint["method"],
            path=path,
            params=query_params if query_params else None,
            data=body_params if body_params else None,
            headers=header_params if header_params else None
        )
    
    {% endif %}
    {% if not discovery_only %}
    {% for tool in tools %}
    
    @mcp.tool()
    async def {{ tool.name }}(
        {%- set path_params = tool.parameters | selectattr('location', 'equalto', 'path') | list -%}
        {%- set query_params = tool.parameters | selectattr('location', 'equalto', 'query') | list -%}
        {%- set header_params = tool.headers -%}
        {%- set body_params = tool.request_body_params -%}
        
        {%- set required_path = path_params | selectattr('required', 'equalto', true) | list -%}
        {%- set required_query = query_params | selectattr('required', 'equalto', true) | list -%}
        {%- set required_body = body_params | selectattr('required', 'equalto', true) | list -%}
        
        {%- set optional_path = path_params | selectattr('required', 'equalto', false) | list -%}
        {%- set optional_query = query_params | selectattr('required', 'equalto', false) | list -%}
        {%- set optional_body = body_params | selectattr('required', 'equalto', false) | list -%}
        
        {%- set all_sorted = required_path + required_query + required_body + optional_path + optional_query + optional_body + header_params -%}
        
        {%- for param in all_sorted %}
        {{ param.name | replace(' ', '_') | replace('-', '_') | lower }}: {%- if param.required and param not in header_params %} {{ param.python_type }}{%- else %} Optional[{{ param.python_type }}] = None{%- endif %}
        {%- if not loop.last %},{% endif %}
        {%- endfor %}
    ) -> Dict[str, Any]:
        """
        {{ tool.description }}
        
        Generated from: {{ tool.method }} {{ tool.path }}
        Source: {{ spec_version }}
        {% if tool.operation_id %}Operation ID: {{ tool.operation_id }}{% endif %}
        {% if tool.deprecated %}⚠️ DEPRECATED: This endpoint is marked as deprecated.{% endif %}
        
        {% if tool.auth_required %}Authentication: Required{% else %}Authentication: Not required{% endif %}
        
        Parameters:
        {% for param in tool.parameters + tool.request_body_params -%}
        - {{ param.name }}: {{ param.openapi_type }}{% if param.required %} (required){% endif %}
        {%- if param.description %}
          {{ param.description }}{% endif %}
        {%- if param.example is not none %}
          Example: {{ param.example | tojson }}{% endif %}
        {%- if param.enum %}
          Allowed values: {{ param.enum | join(', ') }}{% endif %}
        {%- if param.pattern %}
          Pattern: {{ param.pattern }}{% endif %}
        {%- if param.minimum is not none or param.maximum is not none %}
          Range: {% if param.minimum is not none %}{{ param.minimum }}{% else %}-∞{% endif %} to {% if param.maximum is not none %}{{ param.maximum }}{% else %}∞{% endif %}{% endif %}
        {% endfor %}
        
        {% if tool.headers %}
        Headers:
        {% for header in tool.headers -%}
        - {{ header.name }}: {{ header.openapi_type }}
        {%- if header.description %}
          {{ header.description }}{% endif %}
        {% endfor %}
        {% endif %}
        """
        
        {% if tool.request_body_params %}
        # Prepare request data
        {% if tool.request_body_params | length == 1 and tool.request_body_params[0].name == 'request_body' %}
        # Single body parameter - send directly
        data = {{ tool.request_body_params[0].name | replace(' ', '_') | replace('-', '_') | lower }} if {{ tool.request_body_params[0].name | replace(' ', '_') | replace('-', '_') | lower }} is not None else {}
        {% else %}
        # Multiple body parameters - build object
        data = {}
        {% for param in tool.request_body_params -%}
        if {{ param.name | replace(' ', '_') | replace('-', '_') | lower }} is not None:
            {%- if param.discriminator %}
            # Handle discriminated union
            if isinstance({{ param.name | replace(' ', '_') | replace('-', '_') | lower }}, dict) and "{{ param.discriminator }}" not in {{ param.name | replace(' ', '_') | replace('-', '_') | lower }}:
                logger.warning("Discriminator '{{ param.discriminator }}' not found in {{ param.name }}")
            {%- endif %}
            data["{{ param.name }}"] = {{ param.name | replace(' ', '_') | replace('-', '_') | lower }}
        {% endfor %}
        {% endif %}
        
        {% endif %}
        # Prepare query parameters
        params = {}
        {% for param in query_params -%}
        if {{ param.name | replace(' ', '_') | replace('-', '_') | lower }} is not None:
            {% if param.openapi_type == 'array' and param.collection_format -%}
            # Handle array with collection format: {{ param.collection_format }}
            {% if param.collection_format == 'multi' -%}
            # For multi, we'll need to handle this specially in the request
            params["{{ param.name }}"] = {{ param.name | replace(' ', '_') | replace('-', '_') | lower }}
            {% elif param.collection_format == 'csv' -%}
            params["{{ param.name }}"] = ",".join(str(v) for v in {{ param.name | replace(' ', '_') | replace('-', '_') | lower }})
            {% elif param.collection_format == 'ssv' -%}
            params["{{ param.name }}"] = " ".join(str(v) for v in {{ param.name | replace(' ', '_') | replace('-', '_') | lower }})
            {% elif param.collection_format == 'tsv' -%}
            params["{{ param.name }}"] = "\t".join(str(v) for v in {{ param.name | replace(' ', '_') | replace('-', '_') | lower }})
            {% elif param.collection_format == 'pipes' -%}
            params["{{ param.name }}"] = "|".join(str(v) for v in {{ param.name | replace(' ', '_') | replace('-', '_') | lower }})
            {% else -%}
            params["{{ param.name }}"] = {{ param.name | replace(' ', '_') | replace('-', '_') | lower }}
            {% endif -%}
            {% else -%}
            params["{{ param.name }}"] = {{ param.name | replace(' ', '_') | replace('-', '_') | lower }}
            {% endif %}
        {% endfor %}
        
        # Replace path parameters in URL
        path = "{{ tool.path }}"
        {% for param in path_params -%}
        if {{ param.name | replace(' ', '_') | replace('-', '_') | lower }} is not None:
            path = path.replace("{{ '{' }}{{ param.name }}{{ '}' }}", str({{ param.name | replace(' ', '_') | replace('-', '_') | lower }}))
        {% endfor %}
        
        # Prepare headers
        headers = {}
        {% for header in header_params -%}
        if {{ header.name | replace(' ', '_') | replace('-', '_') | lower }} is not None:
            headers["{{ header.name }}"] = str({{ header.name | replace(' ', '_') | replace('-', '_') | lower }})
        {% endfor %}
        
        return await make_api_request(
            "{{ tool.method }}", 
            path, 
            {% if tool.request_body_params %}data=data, {% endif %}
            params=params if params else None, 
            headers=headers if headers else None
        )
    
    {% endfor %}
    {% endif %}
    
    logger.info("Registered {% if include_discovery %}3 discovery{% if not discovery_only %} + {{ tools | length }} typed{% endif %}{% else %}{{ tools | length }}{% endif %} API tools")


# Tool summary for reference
TOOL_SUMMARY = {
    "total_tools": {% if discovery_only %}3{% elif include_discovery %}3 + {{ tools | length }}{% else %}{{ tools | length }}{% endif %},
    {% if include_discovery %}"discovery_tools": ["list_api_endpoints", "get_api_endpoint_schema", "invoke_api_endpoint"],{% endif %}
    {% if not discovery_only %}"typed_tools": {{ tools | length }},{% endif %}
    "catalog_size": {{ tools | length }},
    "by_method": {
        {% for method in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'] -%}
        "{{ method }}": {{ tools | selectattr('method', 'equalto', method) | list | length }}{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    {% if tools_by_tag %}"by_tag": {{ '{' }}{% for tag, tag_tools in tools_by_tag.items() %}"{{ tag }}": {{ tag_tools | length }}{% if not loop.last %}, {% endif %}{% endfor %}{{ '}' }},{% endif %}
    "by_source": {
        "{{ spec_version }}": {{ tools | length }}
    },
    "mode": {% if discovery_only %}"discovery_only"{% elif include_discovery %}"full"{% else %}"typed_only"{% endif %}
}

if __name__ == "__main__":
    print("Tool Summary:")
    print(json.dumps(TOOL_SUMMARY, indent=2))
'''


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Generate FastMCP tools from OpenAPI/Swagger specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from local file (includes discovery tools by default)
  python openapi2mcp.py --spec api.yaml --output mcp_tools.py
  
  # Generate from URL with discovery tools only
  python openapi2mcp.py --url https://api.example.com/openapi.json --output mcp_tools.py --discovery-only
  
  # Generate typed tools only (no discovery)
  python openapi2mcp.py --spec swagger.json --output tools.py --no-discovery
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--spec", 
        type=str, 
        help="Path to OpenAPI/Swagger specification file"
    )
    input_group.add_argument(
        "--url", 
        type=str, 
        help="URL to OpenAPI/Swagger specification"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output Python file path"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        help="Custom server name (defaults to API title from spec)"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (even more verbose)"
    )
    
    # Discovery mode options - these are mutually exclusive
    discovery_group = parser.add_mutually_exclusive_group()
    discovery_group.add_argument(
        "--no-discovery",
        action="store_true",
        help="Disable discovery tools and only generate typed tools"
    )
    discovery_group.add_argument(
        "--discovery-only",
        action="store_true",
        help="Generate only discovery tools (list, schema, invoke) without typed endpoint functions"
    )
    
    # Header fixing options
    parser.add_argument(
        "--auto-fix-headers",
        action="store_true",
        help="Automatically move suspected header parameters from query to header location"
    )
    
    parser.add_argument(
        "--header-patterns",
        type=str,
        nargs='+',
        help="Additional patterns to identify header parameters"
    )
    
    args = parser.parse_args()
    
    # Configure logging level based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        # For normal mode, show INFO for the main script but WARNING for others
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Always show the startup message
    print("🚀 Starting OpenAPI to MCP Generator...")
    
    try:
        # Initialize parser
        spec_path = args.url if args.url else args.spec
        logger.info(f"Loading specification from: {spec_path}")
        
        parser = ComprehensiveOpenAPIParser(
            spec_path, 
            auto_fix_headers=args.auto_fix_headers,
            additional_header_patterns=args.header_patterns
        )
        
        # Parse all tools
        logger.info("Parsing API paths and operations...")
        tools = parser.parse_paths()
        
        if not tools:
            logger.warning("No tools found in specification")
            print("❌ No API operations found in the specification")
            return 1
        
        logger.info(f"Found {len(tools)} operations to convert")
        
        # Determine discovery mode based on arguments
        if args.discovery_only:
            # Discovery-only mode: only generate 3 discovery tools
            include_discovery = True
            discovery_only = True
            mode_desc = "discovery-only"
        elif args.no_discovery:
            # No discovery: only generate typed tools
            include_discovery = False
            discovery_only = False
            mode_desc = "typed-only"
        else:
            # Default: generate both discovery and typed tools
            include_discovery = True
            discovery_only = False
            mode_desc = "full (discovery + typed)"
        
        # Generate output
        logger.info(f"Generating MCP tools file: {args.output} (mode: {mode_desc})")
        
        generator = ComprehensiveMCPToolGenerator(parser)
        generator.generate_tools_file(
            tools, 
            args.output, 
            args.name, 
            include_discovery, 
            discovery_only
        )
        
        # Show summary based on mode
        if discovery_only:
            print(f"✅ Discovery-only mode: Generated 3 discovery tools")
            print(f"   Catalog contains {len(tools)} endpoints from {parser.api_title}")
        elif include_discovery:
            print(f"✅ Full mode: Generated {len(tools)} typed tools + 3 discovery tools")
        else:
            print(f"✅ Typed-only mode: Generated {len(tools)} typed tools (no discovery)")
        
        print(f"\nAPI: {parser.api_title} v{parser.api_version}")
        print(f"Base URL: {parser.base_url}")
        
        # Count by method
        method_counts = {}
        for tool in tools:
            method_counts[tool.method] = method_counts.get(tool.method, 0) + 1
        
        print(f"\nEndpoints by method:")
        for method, count in sorted(method_counts.items()):
            print(f"  {method}: {count}")
        
        # Count by tag
        tag_counts = {}
        for tool in tools:
            for tag in tool.tags or ['untagged']:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        if tag_counts:
            print(f"\nEndpoints by tag:")
            for tag, count in sorted(tag_counts.items()):
                print(f"  {tag}: {count}")
        
        # Show deprecated tools if any
        deprecated_count = sum(1 for tool in tools if tool.deprecated)
        if deprecated_count:
            print(f"\n⚠️  {deprecated_count} deprecated endpoints found")
        
        # Show parameter fixes if any were applied
        if args.auto_fix_headers:
            print(f"\n✅ Auto-fix headers was enabled")
        
        print(f"\nOutput written to: {args.output}")
        
        # Show usage hint based on mode
        if discovery_only:
            print("\nUsage example:")
            print("  from generated_tools import list_api_endpoints, get_api_endpoint_schema, invoke_api_endpoint")
            print("  endpoints = await list_api_endpoints()")
        elif not include_discovery:
            print("\nUsage example:")
            print("  from generated_tools import <endpoint_function_name>")
            print("  result = await <endpoint_function_name>(param1='value1')")
        else:
            print("\nUsage example:")
            print("  # Use discovery tools:")
            print("  from generated_tools import list_api_endpoints")
            print("  endpoints = await list_api_endpoints()")
            print("  # Or use typed functions directly:")
            print("  from generated_tools import <endpoint_function_name>")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Failed to generate tools: {e}")
        if args.debug or args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())