#python3 -m homework.train -log log1 -l 0.1 -o 0.9 -w 0.0001 -e 20 -b 128
python3 -m homework.train -log log -l 0.1 -o 0.99 -w 0.0001 -e 50 -b 512 -d data # train accuracy: 0.87, valid accuracy : 0.84, num_epoch : 41

#python3 -m homework.train -log log3 -l 0.1 -o 0.999 -w 0.0001 -e 20 -b 128
#python3 -m homework.train -log log5 -l 0.05 -o 0.99 -w 0.0001 -e 10 -b 128


python3 -m homework.viz_prediction data/train # Segmentation fault: 11
