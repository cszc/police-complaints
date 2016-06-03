from upload_csv import import_csv

TO_UPLOAD = ["sanitation","vacantbuildings","streetlights_all","vehicles","streetlights_one","treetrims","potholes"]

if __name__=="__main__":
	for new_table in TO_UPLOAD:
		fname = "data/311/"+new_table+".csv"
		import_csv(new_table,fname)
		print("loaded {}".format(new_table))