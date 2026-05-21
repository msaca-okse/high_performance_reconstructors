#!/bin/bash
# Submit x-ray preprocessing, then reconstruction on completion.
# Run from repo root: bash x_ray/jobscripts/submit_all.sh

JOB1=$(bsub < x_ray/jobscripts/jobscript_preprocessor.sh | awk '{print $2}' | tr -d '<>')
bsub -w "done(${JOB1})" < x_ray/jobscripts/jobscript_reconstruction.sh
