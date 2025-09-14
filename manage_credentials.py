#!/usr/bin/env python3
"""
Credential management utility for yt-dl-sub.

This utility provides a command-line interface for managing service credentials
across different profiles (personal, work, client, etc.).
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from core.credential_vault import CredentialVault, ServiceCategory, get_credential_vault
from core.service_credentials import validate_all_credentials


@click.group()
@click.option('--vault-path', type=click.Path(), help='Path to credential vault file')
@click.option('--profile', help='Profile to use')
@click.pass_context
def cli(ctx, vault_path: Optional[str], profile: Optional[str]):
    """Manage credentials for yt-dl-sub services."""
    vault_path = Path(vault_path) if vault_path else None
    ctx.obj = get_credential_vault(vault_path, profile)


@cli.command()
@click.pass_obj
def list_profiles(vault: CredentialVault):
    """List all available credential profiles."""
    profiles = vault.list_profiles()
    current = vault.profile
    
    click.echo("Available profiles:")
    for profile in profiles:
        marker = " (current)" if profile == current else ""
        click.echo(f"  - {profile}{marker}")


@cli.command()
@click.argument('profile_name')
@click.pass_obj
def switch_profile(vault: CredentialVault, profile_name: str):
    """Switch to a different credential profile."""
    try:
        vault.switch_profile(profile_name)
        click.echo(f"Switched to profile: {profile_name}")
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('profile_name')
@click.option('--copy-from', help='Profile to copy credentials from')
@click.pass_obj
def create_profile(vault: CredentialVault, profile_name: str, copy_from: Optional[str]):
    """Create a new credential profile."""
    try:
        vault.create_profile(profile_name, copy_from)
        click.echo(f"Created profile: {profile_name}")
        if copy_from:
            click.echo(f"  Copied from: {copy_from}")
    except (ValueError, KeyError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('profile_name')
@click.confirmation_option(prompt='Are you sure you want to delete this profile?')
@click.pass_obj
def delete_profile(vault: CredentialVault, profile_name: str):
    """Delete a credential profile."""
    try:
        vault.delete_profile(profile_name)
        click.echo(f"Deleted profile: {profile_name}")
    except (ValueError, KeyError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('category', type=click.Choice(['storage', 'ai_text', 'ai_image', 'ai_video', 'ai_audio', 'all']))
@click.pass_obj
def list_services(vault: CredentialVault, category: str):
    """List available services."""
    if category == 'all':
        services = vault.list_services()
    else:
        services = vault.list_services(category)
    
    for cat, service_list in services.items():
        click.echo(f"\n{cat}:")
        for service in service_list:
            click.echo(f"  - {service}")


@cli.command()
@click.argument('category', type=click.Choice(['storage', 'ai_text', 'ai_image', 'ai_video', 'ai_audio']))
@click.argument('service_name')
@click.pass_obj
def get_credentials(vault: CredentialVault, category: str, service_name: str):
    """Get credentials for a specific service."""
    try:
        creds = vault.get_credentials(category, service_name)
        if creds:
            click.echo(f"\nCredentials for {service_name} ({category}):")
            for key, value in creds.items():
                # Mask sensitive values
                if 'key' in key.lower() or 'secret' in key.lower() or 'token' in key.lower():
                    display_value = value[:4] + "..." + value[-4:] if value and len(value) > 8 else "***"
                else:
                    display_value = value
                click.echo(f"  {key}: {display_value}")
        else:
            click.echo(f"No credentials found for {service_name}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('category', type=click.Choice(['storage', 'ai_text', 'ai_image', 'ai_video', 'ai_audio']))
@click.argument('service_name')
@click.option('--profile', help='Profile to update (default: current)')
@click.pass_obj
def set_credentials(vault: CredentialVault, category: str, service_name: str, profile: Optional[str]):
    """Set credentials for a specific service (interactive)."""
    click.echo(f"\nSetting credentials for {service_name} ({category})")
    click.echo("Enter credential values (press Enter to skip):\n")
    
    # Get common fields for the service type
    field_prompts = {
        'storage': {
            'gdrive': ['credentials_file', 'folder_id', 'shared_drive_id'],
            'airtable': ['api_key', 'base_id', 'table_name'],
            's3': ['access_key_id', 'secret_access_key', 'bucket_name', 'region']
        },
        'ai_text': {
            'claude': ['api_key', 'model', 'max_tokens'],
            'openai': ['api_key', 'organization', 'model', 'max_tokens'],
            'gemini': ['api_key', 'model', 'max_tokens'],
            'groq': ['api_key', 'model']
        },
        'ai_image': {
            'dalle': ['api_key', 'model', 'size', 'quality'],
            'midjourney': ['api_key', 'webhook_url'],
            'stable_diffusion': ['api_key', 'model', 'steps'],
            'leonardo': ['api_key', 'model_id'],
            'ideogram': ['api_key', 'model']
        },
        'ai_video': {
            'runway': ['api_key', 'model'],
            'pika': ['api_key'],
            'heygen': ['api_key', 'avatar_id']
        },
        'ai_audio': {
            'elevenlabs': ['api_key', 'voice_id', 'model_id'],
            'murf': ['api_key', 'voice_id'],
            'playht': ['api_key', 'user_id', 'voice']
        }
    }
    
    fields = field_prompts.get(category, {}).get(service_name, ['api_key'])
    credentials = {}
    
    for field in fields:
        # Use hidden prompt for sensitive fields
        if 'key' in field.lower() or 'secret' in field.lower() or 'token' in field.lower():
            value = click.prompt(f"  {field}", hide_input=True, default='', show_default=False)
        else:
            value = click.prompt(f"  {field}", default='', show_default=False)
        
        if value:
            credentials[field] = value
    
    if credentials:
        try:
            vault.set_credentials(category, service_name, credentials, profile)
            click.echo(f"\nCredentials saved for {service_name}")
        except Exception as e:
            click.echo(f"Error saving credentials: {e}", err=True)
            sys.exit(1)
    else:
        click.echo("No credentials entered")


@cli.command()
@click.pass_obj
def validate(vault: CredentialVault):
    """Validate all configured credentials."""
    click.echo("Validating credentials...\n")
    
    results = validate_all_credentials()
    
    categories = {}
    for key, valid in results.items():
        category = key.split('.')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append((key.split('.')[1], valid))
    
    for category, services in categories.items():
        click.echo(f"{category}:")
        for service, valid in services:
            status = "✓" if valid else "✗"
            color = "green" if valid else "red"
            click.secho(f"  {status} {service}", fg=color)
        click.echo()


@cli.command()
@click.argument('profile_name')
@click.argument('output_file', type=click.Path())
@click.pass_obj
def export_profile(vault: CredentialVault, profile_name: str, output_file: str):
    """Export a profile to JSON file."""
    try:
        vault.export_profile(profile_name, Path(output_file))
        click.echo(f"Exported profile '{profile_name}' to {output_file}")
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--profile-name', help='Name for imported profile')
@click.pass_obj
def import_profile(vault: CredentialVault, input_file: str, profile_name: Optional[str]):
    """Import a profile from JSON file."""
    try:
        vault.import_profile(Path(input_file), profile_name)
        click.echo(f"Imported profile from {input_file}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_obj
def show_env_overrides(vault: CredentialVault):
    """Show environment variable overrides."""
    if vault._env_overrides:
        click.echo("Environment variable overrides:")
        for service, fields in vault._env_overrides.items():
            click.echo(f"\n  {service}:")
            for field, value in fields.items():
                # Mask sensitive values
                if 'key' in field.lower() or 'secret' in field.lower():
                    display_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                else:
                    display_value = value
                click.echo(f"    {field}: {display_value}")
    else:
        click.echo("No environment variable overrides found")


@cli.command()
def quick_setup():
    """Interactive quick setup for common services."""
    click.echo("=== Quick Credential Setup ===\n")
    
    vault = get_credential_vault()
    
    # Ask about common services
    services = [
        ('storage', 'gdrive', 'Google Drive'),
        ('storage', 'airtable', 'Airtable'),
        ('ai_text', 'claude', 'Claude (Anthropic)'),
        ('ai_text', 'openai', 'OpenAI'),
        ('ai_text', 'gemini', 'Google Gemini')
    ]
    
    for category, service, display_name in services:
        if click.confirm(f"\nSetup {display_name}?", default=False):
            click.echo(f"\nConfiguring {display_name}...")
            
            # Get service-specific fields
            if service == 'gdrive':
                creds = {
                    'credentials_file': click.prompt('  Service account JSON file path'),
                    'folder_id': click.prompt('  Google Drive folder ID')
                }
            elif service == 'airtable':
                creds = {
                    'api_key': click.prompt('  API key', hide_input=True),
                    'base_id': click.prompt('  Base ID'),
                    'table_name': click.prompt('  Table name', default='Videos')
                }
            else:  # AI services
                creds = {
                    'api_key': click.prompt('  API key', hide_input=True)
                }
            
            try:
                vault.set_credentials(category, service, creds)
                click.secho(f"  ✓ {display_name} configured", fg='green')
            except Exception as e:
                click.secho(f"  ✗ Failed to configure {display_name}: {e}", fg='red')
    
    click.echo("\n=== Setup Complete ===")
    click.echo("\nYou can manage credentials with:")
    click.echo("  python manage_credentials.py --help")


if __name__ == '__main__':
    cli()