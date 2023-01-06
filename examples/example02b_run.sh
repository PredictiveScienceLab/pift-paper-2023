# A script that runs everything about example 2.
#
# Author:
#   Ilias Bilionis
#
# Date:
#   1/2/2023
#
if [ ! -f ./example02b ]; then
  echo "You haven't compiled the code or you are not running this script from the examples directory."
  exit 1
fi

if [ -d example02_newb_results ]; then
  rm -rf example02_newb_results
fi

mkdir example02_newb_results

id=0
for n in 40
do
for sigma in 0.01
do
for gamma in 0.0 0.2 0.4 0.6 0.8 1.0
do
  cmd="./example02b $gamma example02b.yml $n $sigma $id"
  echo $cmd
  eval $cmd
  if [ $? -ne 0 ]; then
    echo "*** FAILED ***"
    exit 2
  fi
  prefix=`printf "example02_gamma=%1.2e_n=%d_sigma=%1.2e_%d" $gamma $n $sigma $id`
  mv ${prefix}*.csv example02_newb_results/
  if [ $? -ne 0 ]; then
    echo "*** FAILED ***"
    exit 2
  fi
done
done
done
