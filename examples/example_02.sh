for n in 4 16 32
do
  for s in 1e-2
  do
    for gamma in 0.0 0.2 0.4 0.8 1.0
    do
      python3 examples/example_02.py \
        --beta-start=10.0 \
        --gamma=$gamma \
        --num-samples=1000 \
        --num-warmup=500 \
        --thinning=10 \
        --nr-alpha=0.1 \
        --nr-maxit=200 \
        --nr-tol=1e-3 \
        --sgld-maxit=200 \
        --sgld-fix-it=200 \
        --sgld-alpha=0.01 \
        --sgld-gamma=0.70 \
        --num-observations=$n \
        --sigma=$s \
        --progress-bar
    done
  done
done
