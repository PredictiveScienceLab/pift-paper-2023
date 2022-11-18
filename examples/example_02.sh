for n in 2 4 8 16 32
do
  for s in 1e-4 1e-2 1e-1
  do
    for gamma in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
      python3 examples/example_02.py \
        --beta-start=10.0 \
        --gamma=$gamma \
        --nr-maxit=10 \
        --nr-tol=1e-2 \
        --sgld-maxit=1000 \
        --sgld-fix-it=900 \
        --num-observations=$n \
        --sigma=$s
    done
  done
done
