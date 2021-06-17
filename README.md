
## Repeated Motif Discovery 

Automated pitch contour extraction and pattern exploration

### Installation
requires essentia library

requires python <= 3.8

Install using 

`pip install -e .` 

### Notebooks
Some notebook walkthroughs are available in `notebooks` for the traditions Carnatic Music, Arab Andalusian Music and Hindustani Music (more details within)

Results for these notebooks can be found in `output`. Those corresponding to the submission _Nuttall, T., Plaja, G., Pearson, L., Serra, X.: The Matrix Profile for Motif Discovery in Audio - An Example Application in Carnatic Music. In: 15th International Symposium Computer Music Multidisciplinary Resarch, Tokyo, 2021_ can be found in `output/indian_carnatic`

### General Usage
To run pipeline on a custom audio, alter the notebooks or use the CLI...

#### CLI Usage

Adjust the configuration parameters in (explanation within)

`conf/pipeline/conf.yaml`

`conf/pipeline/exclusion_conf.yaml`

`conf/pipeline/plot_conf.yaml`

And run

`python src find-motif`

This will default to use the configurations in `conf/pipeline`, custom configurations can passed using...

`python src find-motif --conf-path <conf_path> --exclusion-conf-path <exclusion_conf_path> --plot-conf-path <plot_conf_path>`

Output of plots and audio will be in the output directory specified in `conf/pipeline/conf.yaml`
