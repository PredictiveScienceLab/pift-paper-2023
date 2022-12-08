for n in 16
do
  for s in 1e-2
  do
    for gamma in 0.0 0.2 0.4 0.6 0.8 1.0
    do
      python3 examples/example_02.py \
        --beta-start=1.0 \
        --gamma=$gamma \
        --num-samples=1000 \
        --num-warmup=100 \
        --thinning=10 \
        --nr-alpha=0.1 \
        --nr-maxit=10 \
        --nr-tol=1e-2 \
        --sgld-maxit=500 \
        --sgld-fix-it=100 \
        --sgld-alpha=0.1 \
        --sgld-gamma=0.70 \
        --num-observations=$n \
        --sigma=$s \
        --progress-bar
    done
  done
done
