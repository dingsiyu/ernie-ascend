##!/bin/bash
#set -x
#
#mpirun hostname
#source ./model_conf
#if [[ ${slurm_train_files_dir:-""} != "" ]];then
#    sh /home/HGCP_Program/software-install/afs_mount/bin/afs_mount.sh NLP_KM_Data NLP_km_2018 `pwd`/data ${slurm_train_files_dir}
#    mkdir -p log
#    ls `pwd`/data > ./log/afs_mount.log
#    rm -rf ./data/logs
#fi
#
#if [[ ${slurm_train_packages_dir:-""} != "" ]];then
#    sh /home/HGCP_Program/software-install/afs_mount/bin/afs_mount.sh NLP_KM_Data NLP_km_2018 `pwd`/packages ${slurm_train_packages_dir}
#    ls `pwd`/packages > ./log/afs_mount2.log
#    rm -rf ./packages/logs
#fi
#
#mpirun sh ./slurm/setup.sh
#
#iplist=`cat nodelist-${SLURM_JOB_ID} | xargs  | sed 's/ /,/g'`
#mpirun --bind-to none -x iplist=${iplist} sh train.sh

#!/usr/bin/env bash
source ./model_conf
source ./slurm/utils.sh
#mpirun work with root on k8s
#source ~/.bashrc
init_mpi_env
init_afs_env
set -x
#Only for slurm environment
if [[ ${mpi_on_k8s:-0} != "1" ]];then
    init_slurm_env
fi
if [[ ${slurm_train_files_dir:-""} != "" ]];then
    sh slurm/afs_mount.sh NLP_KM_Data NLP_km_2018 `pwd`/data ${slurm_train_files_dir}
    mkdir -p log
    ls `pwd`/data > ./log/afs_mount.log
fi
if [[ ${remote_package_files_dir:-""} != "" ]];then
    sh slurm/afs_mount.sh NLP_KM_Data NLP_km_2018 `pwd`/packages ${remote_package_files_dir}
    mkdir -p log
    ls `pwd`/packages > ./log/afs_mount.log
fi

#if [[ ${slurm_train_packages_dir:-""} != "" ]];then
#    sh slurm/afs_mount.sh NLP_KM_Data NLP_km_2018 `pwd`/packages ${slurm_train_packages_dir}
#    mkdir -p log
#    ls `pwd`/packages > ./log/afs_mount2.log
#fi
${MPIRUN} -pernode --bind-to none sh ./slurm/setup.sh
RAND_SEED=$RANDOM
${MPIRUN} -pernode --bind-to none -x iplist=${MPI_IPLIST} sh train.sh ${RAND_SEED}
