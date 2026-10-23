#!/bin/bash

module load miniconda/py39_23.1
source /projects/lindorff_group/people/thd711/ensemble_generator_projectPeptone/software/DEER_PREdict/DEERPREdict/bin/activate

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

#cp $code_path/calc_PRE.py ./

python calc_PRE.py

#rm calc_PRE.py

echo "Done with all!"

deactivate

