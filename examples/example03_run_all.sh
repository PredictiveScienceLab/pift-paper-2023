# Run everything about example 3
for s in 0.01 0.0001
do
for n in 10 80
do
  for b in 1000 10000 100000
  do
    ./example03 $b example03.yml $n $s 0 &
  done
  wait
  for b in 1000 10000 100000
  do
    ./example03b $b example03b.yml $n $s 0 &
  done
  wait
done
done
