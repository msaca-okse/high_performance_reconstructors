#!/bin/bash
# Submit neutron preprocessing, then reconstruction on completion.
# Run from repo root: bash neutron/jobscripts/submit_all.sh

JOB1=$(bsub < neutron/jobscripts/jobscript_preprocessor.sh | awk '{print $2}' | tr -d '<>')
bsub -w "done(${JOB1})" < neutron/jobscripts/jobscript_reconstruction.sh
