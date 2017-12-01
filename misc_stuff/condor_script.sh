universe = vanilla 
Initialdir = /scratch/cluster/subhamg/Projects/project/3d-generic/
Executable = /lusr/bin/bash
Arguments = /scratch/cluster/subhamg/Projects/my_script.sh
+Group   = "GRAD"
+Project = "INSTRUCTIONAL"
+ProjectDescription = "Course project for CS394N"
Requirements = TARGET.GPUSlot
getenv = True
request_GPUs = 1
+GPUJob = true 
Log = /scratch/cluster/subhamg/Projects/condor.log
Error = /scratch/cluster/subhamg/Projects/condor.err
Output = /scratch/cluster/subhamg/Projects/condor.out
Notification = complete
Notify_user = subhamg@cs.utexas.edu
Queue 1