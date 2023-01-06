for n in 10 20 40 80
do
for sigma in 0.01 0.0001
do
  python3 example02_generate_data.py --num-observations=$n --sigma=$sigma
done
done
