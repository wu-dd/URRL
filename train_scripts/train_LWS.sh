log_path='/mnt/nfs-storage-personal/log/LWS/LWS_eid'
loss='lws'
cd /mnt/nfs-storage-personal/UPLLRS-FULL/

# ==========================================CIFAR-10==========================================
#eid1-5 cifar10 0.1 0.1
pr=0.1
nr=0.1
e_s=1
e_e=5
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar10 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid6-10 cifar10 0.1 0.3
pr=0.1
nr=0.3
e_s=6
e_e=10
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar10 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid11-15 cifar10 0.1 0.5
pr=0.1
nr=0.5
e_s=11
e_e=15
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar10 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid16-20 cifar10 0.3 0.1
pr=0.3
nr=0.1
e_s=16
e_e=20
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar10 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid21-25 cifar10 0.3 0.3
pr=0.3
nr=0.3
e_s=21
e_e=25
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar10 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid26-30 cifar10 0.3 0.5
pr=0.3
nr=0.5
e_s=26
e_e=30
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar10 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid31-35 cifar10 0.5 0.1
pr=0.5
nr=0.1
e_s=31
e_e=35
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar10 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid36-40 cifar10 0.5 0.3
pr=0.5
nr=0.3
e_s=36
e_e=40
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar10 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid41-45 cifar10 0.5 0.5
pr=0.5
nr=0.5
e_s=41
e_e=45
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar10 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

# ==========================================CIFAR-100==========================================
#eid101-105 cifar100 0.01 0.1
pr=0.01
nr=0.1
e_s=101
e_e=105
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar100 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid106-110 cifar100 0.01 0.3
pr=0.01
nr=0.3
e_s=106
e_e=110
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar100 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid111-115 cifar100 0.01 0.5
pr=0.01
nr=0.5
e_s=111
e_e=115
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar100 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid116-120 cifar100 0.05 0.1
pr=0.05
nr=0.1
e_s=116
e_e=120
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar100 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid121-125 cifar100 0.05 0.3
pr=0.05
nr=0.3
e_s=121
e_e=125
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar100 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid126-130 cifar100 0.05 0.5
pr=0.05
nr=0.5
e_s=126
e_e=130
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar100 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid131-135 cifar100 0.1 0.1
pr=0.1
nr=0.1
e_s=131
e_e=135
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar100 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid136-140 cifar100 0.1 0.3
pr=0.1
nr=0.3
e_s=136
e_e=140
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar100 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done

#eid141-145 cifar10 0.1 0.5
pr=0.1
nr=0.5
e_s=141
e_e=145
for eid in `seq $e_s $e_e`;do
    python train.py -eid ${eid} -loss $loss -dataset cifar100 -partial_rate $pr -noisy_rate $nr -exclusion none -seed `expr ${eid} - ${e_s}` > $log_path$eid".log" 2>&1
done