setwd("D:/JooGitRepo/Udacity_DataScientistNanoDegree/Chp1_IntroToDataScience/bn")

bn_cate = read.table("./raw/bn_cate.txt",sep="\t",header=T,row.names=1,check.names=F)
bn_num = read.table("./raw/bn_num.txt",sep="\t",header=T,row.names=1)

node = c(colnames(bn_num),colnames(bn_cate))
write.table(bn_num,"bn_con.txt",sep="\t",row.names=F,col.names=F)

# read in matlab result
bn_num_dis = read.table("dis_varpar_bn_con.txt",sep="\t")
dis_data = cbind(bn_num_dis ,bn_cate)

# output result
write.table(dis_data,"bn_seattle.txt",sep="\t",row.names=F,col.names=F)
write.table(node,"node.txt",sep="\t",row.names=F,col.names=F,quote=F)
write.table(cbind(node,node),"pc.txt",sep="\t",row.names=F,col.names=F,quote=F)
write.table(cbind(node,3),"nodenomial.txt",sep="\t",row.names=F,col.names=F,quote=F)
write.table(matrix(0,80,80),"seedMatrix.txt",sep="\t",row.names=F,col.names=F)

