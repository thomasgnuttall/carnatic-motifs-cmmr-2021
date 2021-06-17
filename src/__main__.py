import click

from src.core import find_motifs
from src.io import load_yaml

@click.group()
def cli():
    pass


@cli.command(name="find-motifs")
@click.option('--conf-path', type=str, default='conf/pipeline/conf.yaml', required=False)
@click.option('--exclusion-conf-path', type=str, default='conf/pipeline/exclusion_conf.yaml', required=False)
@click.option('--plot-conf-path', type=str, default='conf/pipeline/plot_conf.yaml', required=False)
def cmd_find_motifs(conf_path, exclusion_conf_path, plot_conf_path):
    """
    Run motif finding pipeline,
        Extract pitch track
        Find motifs in matrix profile
        Output plots and audio
    """
    print('yes')
    gen_conf = load_yaml(conf_path)
    exclusion_conf = load_yaml(exclusion_conf_path)
    plot_conf = load_yaml(plot_conf_path)

    find_motifs(gen_conf, exclusion_conf, plot_conf)


if __name__ == '__main__':
    cli()