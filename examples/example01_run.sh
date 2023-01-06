# A script the runs everything about example 1.
#
# You must have first compiled the code using make.
# Then you must cd in the examples directory and run this script by:
#
# ./example01_run.sh
#
# The results are written in the director example01_results.
# If the directory already exists, it is deleted and recreated.
#
# Author:
#   Ilias Bilionis
#
# Date:
#   12/21/2022
#

echo "*** START EXAMPLE 01 ***"

if [ ! -f ./example01 ]; then
  echo "You haven't compiled the code or you are not running this script from the examples directory."
  exit 1
fi

for beta in 100.0 1000.0 10000.0 100000.0
do
  echo ">>> RUNNING AT BETA=$beta"
  ./example01 $beta example01.yml
  if [ $? -ne 0 ]; then
    echo "*** FAILED ***"
    exit 2
  fi
done

# Move all datafiles in example01_resulsts
if [ -d example01_results ]; then
  rm -rf example01_results
fi
mkdir example01_results
mv example01*.csv example01_results/
mv example01*.png example01_results/

# Make all plots
python3 example01_make_plots.py

echo "*** END EXAMPLE 01 ***"
