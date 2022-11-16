for beta in 0.0 1.0 10.0 100.0 1000.0 10000.0
do
  python3 examples/example_01.py --beta=${beta} \
    --num-warmup=10000 \
    --num-samples=20000 \
    --thinning=1000
done
