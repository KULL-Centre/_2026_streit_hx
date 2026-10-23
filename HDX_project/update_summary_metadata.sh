#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

cp $code_path/collect_metadata_summary.py ./

python collect_metadata_summary.py

rm collect_metadata_summary.py

echo "Metadata summary is now updated"

conda deactivate

