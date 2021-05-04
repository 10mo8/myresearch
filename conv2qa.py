import re

#質問と回答の関連付けを行います。
def get_question(finname, foutname):
    with open(finname, "r", encoding="utf-8") as fin, open(foutname, "w", encoding="utf-8") as fout:
        texts = fin.readlines()
        change_lst = []
        pre_name = ""
        for i, text in enumerate(texts[:100]):
            name, text = text.split(",")
            #名前が切り替わったタイミングを保存する。初めの方は例外なのでいらない。
            #切り替わる前の発言を取りたいので-1
            if name != pre_name and i != 0 and i != 1:
                change_lst.append(i-1)
            pre_name = name
        
        qcand_lst = []
        check_words = ["お伺い", "いただければ", "答弁"]
        for i in change_lst:
            print(texts[i])
            if re.match(r"(.+お伺い|.+いただければ|.+答弁)", texts[i]):
                qcand_lst.append(i)
                #print(texts[i])
        
        for cand in qcand_lst:
            atten_idx = change_lst.index(cand)
            question = texts[cand]
            question = question.replace("\n", "")
            que = question.split(",")[1]
    
            answer_lst = texts[change_lst[atten_idx]:change_lst[atten_idx+1]]
            ans = ""
            for answer in answer_lst:
                answer = answer.replace("\n", "")
                answer = answer.split(",")[1]
                ans += answer

            print("question", que)
            print("answer", ans)
            fout.write(que + "," + ans + "\n")
        print(change_lst)
        print(qcand_lst)
        

if __name__ == "__main__":
    finname = "./data/cln201bud_com.txt"
    foutname = "./data/qapair.csv"
    get_question(finname, foutname)