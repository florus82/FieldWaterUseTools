#!/bin/bash

dforce() {
    docker run \
    -e FORCE_CREDENTIALS=/app/credentials \
    -e BOTO_CONFIG=/app/credentials/.boto \
    -v $HOME:/app/credentials \
    -v /data:/data \
    -v /mnt:/mnt \
    -v $HOME:/home/potzschf \
    -w $PWD \
    -u $(id -u):$(id -g) \
    -it \
    --rm \
    davidfrantz/force:latest "$@"
}

echo "docker success"

run_for_loop() {
    cd /data/Aldhani/eoagritwin/force/parameterfile/for_Thuenen/S2/magic_subs
    for p in *.prm; do
        dforce force-higher-level "$p"
    done
}


while true; do
    # Run Python script and capture output
    result=$(/data/Aldhani/users/potzschf/conda/envs/workhorse/bin/python ~/repos/fields/Force/py/FORCE_clean_up_Update_prm_FBM.py)
    echo "Python script returned: $result"

    if [[ "$result" == "continue" ]]; then
        echo "Running the single dforce command before the loop..."
        
	dforce force-magic-parameters -c paired /data/Aldhani/eoagritwin/force/parameterfile/for_Thuenen/S2/2017_FBM_magic_S2_cleanUp.prm -o /data/Aldhani/eoagritwin/force/parameterfile/for_Thuenen/S2/magic_subs/


        run_for_loop

        # loop repeats, runs python script again
    elif [[ "$result" == "finished" ]]; then
        echo "Finished signal received. Exiting."
        exit 0
    else
        echo "Unexpected output from python script: $result"
        exit 1
    fi
done

