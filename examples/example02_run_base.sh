# A script that reproduces Examples 2.a and 2.b.
#
# Author:
#   Ilias Bilionis
#
# Date:
#   1/2/2023
#
example02_exec="example02$1"
if [ ! -f ./example02a ]; then
  echo "You haven't compiled the code or you are not running this script from the examples directory."
  exit 1
fi

out_folder=${example02_exec}_results
if [ -d $out_folder ]; then
  rm -rf $out_folder
fi

mkdir $out_folder

id=0
n=40
sigma=0.01
for gamma in 0.0 0.2 0.4 0.6 0.8 1.0
do
  cmd="./$example02_exec $gamma ${example02_exec}.yml $n $sigma $id"
  echo $cmd
  eval $cmd
  if [ $? -ne 0 ]; then
    echo "*** FAILED ***"
    exit 2
  fi
  prefix=`printf "example02_gamma=%1.2e_n=%d_sigma=%1.2e_%d" $gamma $n $sigma $id`
  mv ${prefix}*.csv $out_folder/
  if [ $? -ne 0 ]; then
    echo "*** FAILED ***"
    exit 2
  fi
done

# Make the plots
python3 example02_make_plots.py $n $sigma $out_folder
