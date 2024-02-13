import transform.fuse
import transform.interchange
import transform.parallelize
import transform.split
import transform.cuda_smem


passes = []

fu = fuse.fuser()
fu.register(fuse.basic_rule)
passes.append(fu)