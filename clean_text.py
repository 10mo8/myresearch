import re
#文章を句点ごとに区切るようにして保存します。
def split_period(finname, foutname):
    with open(finname, "r", encoding="utf-8") as fin, open(foutname, "w", encoding="utf-8") as fout:
        texts = fin.read()
        texts = texts.split("。")
        for text in texts:
            text = text.replace("\n", "")
            text = re.sub("^　+", "", text)
            fout.write(text + "。" + "\n")
        
#国会会議録の話者と発言内容を紐づけます。
def link_name(finname, foutname):
    with open(finname, "r", encoding="utf-8") as fin, open(foutname, "w", encoding="utf-8") as fout:
        texts = fin.readlines()
         
        for text in texts:
            match_name = re.search("○.+　", text)
            #名前がある行では名前を保存して、文中の名前を削除しておく。
            if match_name is not None:
                name = match_name.group(0)
                text = re.sub("○.+　", "", text)
            
            #委員長の発言は進行を促すものなのでいらない
            if re.match(".+委員長", name):
                continue
            fout.write(name[1:-1] + "," + text)
                
            
if __name__ == "__main__":
    finname = "./data/201bud_com.txt"
    foutname1 = "./data/spl_peri201bud_com.txt"
    foutname2 = "./data/cln201bud_com.txt"
    split_period(finname, foutname1)
    link_name(foutname1, foutname2)
    