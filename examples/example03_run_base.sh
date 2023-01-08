# Runs example 3 of the paper
# 
# Author:
#   Ilias Bilionis
#
# Date:
#   1/6/2023
#

example03_exec="example03$1"
if [ ! -f $example03_exec ]; then
  echo "You haven't compiled the code or you are not running this script from the examples directory."
  exit 1
fi

out_folder=${example03_exec}_results
if [ -d $out_folder ]; then
  rm -rf $out_folder
fi

mkdir $out_folder

id=0
beta=10000
n=40
sigma=0.01
cmd="./$example03_exec $beta ${example03_exec}.yml $n $sigma $id"
echo $cmd
eval $cmd
if [ $? -ne 0 ]; then
  echo "*** FAILED ***"
  exit 2
fi
prefix=`printf "example03_beta=%1.2e_n=%d_sigma=%1.2e_%d" $beta $n $sigma $id`
mv ${prefix}*.csv $out_folder/
if [ $? -ne 0 ]; then
  echo "*** FAILED ***"
  exit 2
fi

# Make the plots
python3 example03_make_plots.py $beta $n $sigma $1
