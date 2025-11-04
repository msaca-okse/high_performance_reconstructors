JOB1=$(bsub < jobscripts/jobscript_preprocessor.sh | awk '{print $2}' | tr -d '<>')
bsub -w "done(${JOB1})" < jobscripts/jobscript_reconstruction.sh


###### Run with >>> bash submit_all.sh