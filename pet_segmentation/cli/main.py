"""
Main CLI interface for PET Segmentation.
"""

import click
import sys
from pathlib import Path
from typing import Optional, List
import json
import time

from ..config.settings import get_settings
from ..core.segmenter import PETSegmenter
from ..utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version and exit')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help='Set logging level')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, version, debug, log_level, config):
    """PET Segmentation: Automated PET imaging analysis with nnInteractive."""
    
    # Setup context
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['log_level'] = log_level
    ctx.obj['config'] = config
    
    # Setup logging
    if log_level:
        setup_logging(log_level=log_level)
    elif debug:
        setup_logging(log_level='DEBUG')
    
    if version:
        from .. import __version__
        click.echo(f"PET Segmentation version {__version__}")
        return
    
    # If no command specified, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='./outputs',
              help='Output directory for segmentation results')
@click.option('--max-regions', '-m', type=int, default=5,
              help='Maximum number of regions to segment')
@click.option('--server-url', '-s', type=str, default=None,
              help='nnInteractive server URL')
@click.option('--threshold', '-t', type=int, default=90,
              help='Percentile threshold for bright region detection (50-99)')
@click.option('--min-size', type=int, default=100,
              help='Minimum region size in voxels')
@click.option('--format', 'output_format', type=click.Choice(['nifti', 'nifti_gz', 'numpy']),
              default='nifti_gz', help='Output format')
@click.option('--save-intermediate', is_flag=True,
              help='Save intermediate processing results')
@click.option('--quality-threshold', type=float, default=0.5,
              help='Minimum quality score to accept segmentation (0.0-1.0)')
@click.option('--dry-run', is_flag=True, help='Show what would be processed without running')
@click.pass_context
def segment(ctx, input_path, output_dir, max_regions, server_url, threshold, 
           min_size, output_format, save_intermediate, quality_threshold, dry_run):
    """Segment a single PET scan."""
    
    try:
        # Setup
        settings = get_settings()
        if server_url:
            settings.nninteractive_url = server_url
        
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        click.echo(f"🔍 Processing: {input_path}")
        click.echo(f"📂 Output directory: {output_dir}")
        click.echo(f"🎯 Max regions: {max_regions}")
        click.echo(f"🌐 Server: {settings.nninteractive_url}")
        
        if dry_run:
            click.echo("✅ Dry run completed - no actual processing performed")
            return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize segmenter
        with click.progressbar(length=100, label='Initializing') as bar:
            segmenter = PETSegmenter(nninteractive_url=settings.nninteractive_url)
            bar.update(20)
            
            # Test connection
            if not segmenter.test_server_connection():
                click.echo("❌ Failed to connect to nnInteractive server", err=True)
                sys.exit(1)
            bar.update(40)
        
        # Process segmentation
        click.echo("🚀 Starting segmentation...")
        start_time = time.time()
        
        results = segmenter.auto_segment_all_regions(
            dicom_path=str(input_path),
            max_regions=max_regions
        )
        
        processing_time = time.time() - start_time
        
        # Handle results
        if not results.get('success'):
            click.echo(f"❌ Segmentation failed: {results.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
        
        # Display results
        segmentations = results.get('segmentations', [])
        successful_segmentations = [s for s in segmentations if 'segmentation' in s]
        
        click.echo(f"\n✅ Segmentation completed in {processing_time:.2f}s")
        click.echo(f"🔍 Regions detected: {results.get('total_regions_detected', 0)}")
        click.echo(f"🎯 Regions processed: {results.get('regions_processed', 0)}")
        click.echo(f"✅ Successful segmentations: {len(successful_segmentations)}")
        
        # Save results
        saved_files = []
        for seg in successful_segmentations:
            if seg['validation']['quality_score'] >= quality_threshold:
                # Generate output filename
                region_id = seg['region_id']
                timestamp = int(time.time())
                filename = f"region_{region_id}_{input_path.stem}_{timestamp}"
                
                if output_format == 'nifti':
                    output_path = output_dir / f"{filename}.nii"
                elif output_format == 'nifti_gz':
                    output_path = output_dir / f"{filename}.nii.gz"
                else:  # numpy
                    output_path = output_dir / f"{filename}.npy"
                
                # Save segmentation
                segmenter.save_segmentation(seg['segmentation'], str(output_path))
                saved_files.append(output_path)
                
                # Display region info
                validation = seg['validation']
                click.echo(f"  Region {region_id}: Quality {validation['quality_score']:.3f}, "
                          f"Volume {validation['volume']} voxels → {output_path.name}")
        
        # Save metadata
        metadata_file = output_dir / f"segmentation_metadata_{int(time.time())}.json"
        with open(metadata_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        click.echo(f"\n💾 Results saved to: {output_dir}")
        click.echo(f"📊 Metadata saved to: {metadata_file.name}")
        click.echo(f"📁 Segmentation files: {len(saved_files)}")
        
    except Exception as e:
        logger.error("CLI segmentation failed", error=str(e))
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--server-url', '-s', type=str, default=None,
              help='nnInteractive server URL to test')
@click.option('--timeout', type=int, default=10,
              help='Connection timeout in seconds')
def test_connection(server_url, timeout):
    """Test connection to nnInteractive server."""
    
    settings = get_settings()
    url = server_url or settings.nninteractive_url
    
    click.echo(f"🔗 Testing connection to {url}")
    
    try:
        segmenter = PETSegmenter(nninteractive_url=url)
        
        with click.progressbar(length=100, label='Testing connection') as bar:
            bar.update(50)
            success = segmenter.test_server_connection()
            bar.update(100)
        
        if success:
            click.echo("✅ Connection successful!")
        else:
            click.echo("❌ Connection failed!", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Connection error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config.yaml',
              help='Output configuration file path')
def create_config(output):
    """Create a sample configuration file."""
    
    try:
        from ..config.settings import create_config_file
        create_config_file(output)
        click.echo(f"✅ Configuration file created: {output}")
        
    except Exception as e:
        click.echo(f"❌ Failed to create config: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show system information and configuration."""
    
    try:
        settings = get_settings()
        
        click.echo("🏥 PET Segmentation System Information")
        click.echo("=" * 50)
        
        # Version info
        from .. import __version__
        click.echo(f"Version: {__version__}")
        
        # Configuration
        click.echo(f"\nConfiguration:")
        click.echo(f"  nnInteractive URL: {settings.nninteractive_url}")
        click.echo(f"  Debug mode: {settings.debug}")
        click.echo(f"  Log level: {settings.log_level}")
        click.echo(f"  Output directory: {settings.output_dir}")
        click.echo(f"  Output format: {settings.output_format}")
        click.echo(f"  Max regions per image: {settings.max_regions_per_image}")
        click.echo(f"  Min quality score: {settings.min_quality_score}")
        click.echo(f"  Batch size: {settings.batch_size}")
        click.echo(f"  Max workers: {settings.max_workers}")
        
        # System info
        import platform
        import psutil
        
        click.echo(f"\nSystem:")
        click.echo(f"  Platform: {platform.platform()}")
        click.echo(f"  Python: {platform.python_version()}")
        click.echo(f"  CPU cores: {psutil.cpu_count()}")
        click.echo(f"  Memory: {psutil.virtual_memory().total // (1024**3)} GB")
        
        # Dependencies
        click.echo(f"\nKey Dependencies:")
        try:
            import numpy
            click.echo(f"  NumPy: {numpy.__version__}")
        except ImportError:
            click.echo("  NumPy: Not installed")
        
        try:
            import pydicom
            click.echo(f"  PyDICOM: {pydicom.__version__}")
        except ImportError:
            click.echo("  PyDICOM: Not installed")
        
        try:
            import nibabel
            click.echo(f"  NiBabel: {nibabel.__version__}")
        except ImportError:
            click.echo("  NiBabel: Not installed")
        
    except Exception as e:
        click.echo(f"❌ Error getting system info: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main() 