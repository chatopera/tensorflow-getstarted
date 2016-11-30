#! /bin/bash 
###########################################
# Start TensorFlow Board
# http://eng.snaplingo.net/how-to-use-tensorboard/
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
tf_board_port=6006
logdir_root=/tmp/mnist
run_prefix=run
logdir=''

# functions
function generate_logdir(){
    cd $logdir_root
    for x in `ls`;do 
        echo $run_prefix-$x:$logdir_root/$x
        logdir=$run_prefix-$x:$logdir_root/$x,$logdir
    done;
    echo $logdir
}

function start_tensorboard(){
    tensorboard --logdir=$logdir --port $tf_board_port
}

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
generate_logdir
start_tensorboard
