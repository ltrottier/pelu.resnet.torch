for depth in 110
do
for i in 1 2 3 4 5
do
for ds in cifar10 cifar100
do
   savepath=results/results-bn-prelu/try${i}/${ds}/${depth}
   mkdir -p ${savepath}
   th main.lua -netType resnet-bn-prelu -resume ${savepath} -dataset ${ds} -batchSize 256 -nEpochs 200 -depth ${depth} -shortcutType A -weightDecay 0.001 >> ${savepath}/nohup.out

   sleep 1m
done
done
done
