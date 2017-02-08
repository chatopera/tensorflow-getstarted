#! /bin/bash 
###########################################
# Start TensorFlow Board
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
tf_board_port=6006
logdir_root=$baseDir/save
logdir=''

# functions
function generate_logdir(){
    cd $logdir_root
    for x in `ls`;do 
        logdir=$x:$logdir_root/$x,$logdir
    done;
    logdir=${logdir::-1}
    echo "watching " $logdir
}

function start_tensorboard(){
    tensorboard --logdir=$logdir --port $tf_board_port
}

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
generate_logdir
start_tensorboard
