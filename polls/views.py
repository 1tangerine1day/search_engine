from django.shortcuts import render

from .models import UploadFile
from .models import Index
from .models import MeshWord
from .LSTM import LSTMTagger

from django.utils.safestring import mark_safe
from django.core.files.storage import FileSystemStorage

import os
import json
import math
import wikipedia
import torch as torch
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import gensim
from gensim import corpora, models, similarities

import urllib.request
import urllib.parse
from urllib.parse import quote
import urllib.request; #用來建立請求
import re
import string

import threading

lstm_model = LSTMTagger(10,10,19,2)
lstm_model = torch.load('./polls/model_eof.pkl',map_location='cpu')
word_to_ix = {'SPACE': 0, 'ADP':1, 'ADV':2, 'AUX':3, 'CONJ':4, 'CCONJ':5, 'DET':6, 'INTJ':7, 'NUM':8, 'PART':9, 'PRON':10,'PROPN':11, 'PUNCT':12, 'SCONJ':13, 'SYM':14, 'VERB':15, 'NOUN':16, 'X':17, 'ADJ': 18 }
nlp = spacy.load('en_core_web_sm')
jsonDec = json.decoder.JSONDecoder()
model = gensim.models.Word2Vec.load("./polls/pubmed_w2v.model")

# Create your views here.
def home(request):
    return render(request,'homepage.html',)


def upload(request):
    if request.method == 'POST':
        if len(request.FILES) != 0:

            names=[]
            for i in UploadFile.objects.all():
                names.append(i.name)

            #count id
            t = UploadFile.objects.last()
            if t is not None:
                id = t.fid + 1
            else:
                id = 0
            
            #file processing
            test_file = request.FILES['test_file']
            file_name = test_file.name
            file_title,file_text = file_parser(test_file)
            file_eof,_ = e_o_f(file_text)
            file_lemmas = [l.lemma_ for l in nlp(file_text) if l.is_stop is False or l.pos_ not in ("PUNCT","ADP","SPACE","DET","CCONJ")]

            if file_name not in names:
                #file data store
                UploadFile.objects.create(
                    fid = id,
                    name = file_name,
                    title = file_title,
                    text = file_text,
                    lemmas = json.dumps(file_lemmas),
                    eof = json.dumps(file_eof),
                    )

                #update index
                for i in file_lemmas:
                    query_word = Index.objects.filter(lemma = i).last()
                    if query_word:
                        temp1 = query_word.position
                        temp2 = query_word.lemma
                        new_position = jsonDec.decode(query_word.position)
                        if file_name not in new_position : 
                            Index.objects.filter(lemma = i).update(
                                lemma = i,
                                position = json.dumps(new_position + [file_name])
                            )
                    else:
                        Index.objects.create(
                            lemma = i,
                            position = json.dumps([file_name,]),
                            mesh = FindMeshWord(i)
                        )
            else:
                render(request,'homepage.html',)

        return render(request,'homepage.html',)
    else:
        return render(request,'homepage.html',)


def result(request):
    if request.method == 'POST':
        #get search word
        if 'search_word' in request.POST:
            search_word = request.POST['search_word']
            if search_word in (None,""):
                return render(request,'homepage.html',)
        else:
            return render(request,'homepage.html',)

        #show all file
        if search_word == "#all":
            files = UploadFile.objects.all()
            find_file = []
            for f in files:
                find_file+=[( mark_safe("<a href='/file/"+str(f.id)+"/'>"+f.name+"</a>"), f.text)]
            return render(request,'result.html',{
                    'contxt':find_file,
                    'not_found':str(len(files)),
                
                })
            
        #print topic
        elif "#topic" in search_word:
            try:
                num = int(search_word.replace("#topic",""))
            except ValueError:
                num = 5

            corpus = corpora.MmCorpus('./polls/pubmed.mm')
            lsi = models.LsiModel.load('./polls/pubmed.lsi')
            dictionary = corpora.Dictionary.load('./polls/pubmed.dict')

            return_str = []
            for i,j in lsi.show_topics(num_topics=num, num_words=10, formatted=False):
                temp_str = ''
                for x, y in j :
                    temp_str += ' '+ str(x)+','

                return_str.append(('Topic' + str(i+1), temp_str))

            return render(request,'result.html',{
                'not_found':'---TOPIC---',
                'sim':return_str
                })
    
        #search word
        else:
            search_lemma = [l.lemma_ for l in nlp(search_word) if l.is_stop is False or l.pos_ not in ("PUNCT","ADP","SPACE","DET","CCONJ")]
            display_type = request.POST.get("search_type", None)

            ex_search_lemma = search_lemma
            
            mesh_word=[]
            relateMesh=[]
            if display_type in ["search_mesh",]:
                mesh_word = FindMeshWord(search_lemma[0])
                relateMesh = SelectMeshTable(search_lemma[0])
                ex_search_lemma = list(set(search_lemma + relateMesh))

            w2v=[]
            if display_type in ["search_w2v",]:
                for l in search_lemma:
                    try: 
                        result = model.wv.most_similar(l)[:5]
                        for each in result:
                            w2v.append(str(each[0]))
                    except KeyError:
                        print(None)
                ex_search_lemma = ListMerge(ex_search_lemma,w2v)

            wikiPrint=[]
            if display_type in ["search_wiki",]:
                wikiPrint = wikipedia.search(""+search_word)
                for i in wikiPrint[:5]:
                    ex_search_lemma = ListMerge(ex_search_lemma,text_to_lemma_delet_pos(i)) 
          
            
    
            find_lemma = Index.objects.filter(lemma__in = ex_search_lemma)

            if len(find_lemma):
                files=[]
                doc_name=[]
                doc_title=[]
                doc_lemma=[]
                doc_text=[]
            

                find_file = UploadFile.objects.all()
                for i in find_file:
                    doc_lemma.append(jsonDec.decode(i.lemmas))
                    doc_name.append(mark_safe("<a href='/file/"+str(i.id)+"/'>"+i.name +"</a>"))
                    doc_text.append(str(i.text))
                    doc_title.append(str(i.title))

                corpus = corpora.MmCorpus('./polls/pubmed.mm')
                lsi = models.LsiModel.load('./polls/pubmed.lsi')
                dictionary = corpora.Dictionary.load('./polls/pubmed.dict')

                vec_bow = dictionary.doc2bow(ex_search_lemma)
                vec_lsi = lsi[vec_bow]
                index = similarities.MatrixSimilarity(lsi[corpus]) 
                sims = index[vec_lsi] 
                sims = sorted(enumerate(sims), key=lambda item: -item[1])


                return_find = []
                for s in sims[:10]:
                    index = int(s[0])
                    temp =  "相似度：" + str(s[1]) + "<br>" + "文章標題：" + str(doc_title[index]) + "<br>" + "文章內容：" + "<br>&nbsp&nbsp&nbsp&nbsp"+ str(doc_text[index])
                    return_find.append((doc_name[index],mark_safe(temp)))
   
                return render(request,'result.html',{
                    'contxt':return_find,
                    'search_lemma':search_lemma,
                    'mesh_word':mesh_word,
                    'relateMesh':relateMesh,
                    'w2v':w2v,
                    'wiki': wikiPrint,
                    'ex_search_lemma':ex_search_lemma,   
                })
            #search word not in dictionary
            else:
                not_found="------------Cannot found '" + search_word + "'-------------"
                return render(request,'result.html',{
                    'not_found':not_found,
                    })
    
    else:
        return render(request,'homepage.html',)


#file page
def show_file(request,open_id):
    
    file = UploadFile.objects.filter(id=open_id).last()
    
    eof_list = jsonDec.decode(file.eof)

    pos = []
    for s in eof_list:
        pos.append([j.pos_ for j in nlp(s)])

    return render(request,'file.html',{
        'name':file.name,
        'title':file.title,
        'text':file.text,
        'text_lemma':file.lemmas,
        'eof':eof_list,
        'pos':pos,
        })

def updata_model(request):
    search_preprocess()
    return render(request,'homepage.html',)

def crawler(request):
    startThread()
    return render(request,'homepage.html',)

##------------------------------------------------------------------------------------
def file_parser(file):
    text=""
    title=""

    type = os.path.splitext(file.name)[-1]

    #read txt
    if type=='.txt':
        text = str(file.read().decode('utf-8'))
    #read xml
    elif type=='.xml':
        soup = BeautifulSoup(file.read(),"lxml")
        xml_title = soup.find('articletitle').text
        xml_text = soup.find('abstract').find('abstracttext').text
        title = str(xml_title)
        text = str(xml_text)
    #read json
    elif type=='.json':
        data = json.loads(file.read())
        json_text=""
        for d in data:
            json_text += d["Text"]
        text = str(son_text)
    
    return title,text

def input_to_tensor(sentence):
    idx=[]
    doc = nlp(sentence)
    for i in range(len(doc)):
        idx += [ word_to_ix[ doc[i].pos_ ] ]
    return torch.tensor(idx, dtype=torch.long)

def e_o_f(text):
    counter=0 
    output=[]
    input_text = text.split('.',text.count('.'))

    temp_sen=""
    for sen in input_text:
    
        if sen is not "" and " " and "   ":
            lstm_output = lstm_model(input_to_tensor(sen))

            x = lstm_output.data.numpy()[-1,0]
            y = lstm_output.data.numpy()[-1,1]

            if x < y :
                output += [temp_sen + sen + "."]
                temp_sen = ""
                counter += 1 
            else:
                if y > -2.3:
                    output += [temp_sen + sen + "."]
                    temp_sen = ""
                    counter += 1 
                else:
                    temp_sen += sen
                    temp_sen += "."
    
    
    if temp_sen is not "" and " " and "   ":
        output += [sen + "."]

    return output,counter

def text_to_lemma_delet_pos(text):
    temp=[]
    for i in nlp(text):
        if i.pos_ not in ("PUNCT","ADP","SPACE","DET","CCONJ"):
            temp.append(i.lemma_)
    return temp

def ListMerge( list1, list2):
    return list1 + list(set(list2) - set(list1))

def FindMeshWord(keyword):
    url = r"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh;term="
    tmpurl = quote(url+keyword, safe=string.printable)
    x = urllib.request.urlopen(tmpurl)
    response = str(x.read().decode('utf-8','ignore').replace(u'\xa9', u''))
    pat = re.compile("<QueryTranslation>(.+)<\/QueryTranslation>")
    smeshLi = pat.findall(response)
    if len(smeshLi) == 0:
        return [""]
    smeshLi[0] = smeshLi[0].replace('"',"")
    smeshLi = smeshLi[0].split(" OR ")
    return json.dumps(smeshLi)

def SelectMeshTable(meshWord):
    sqlstr = '%'+meshWord+'%'
    result = []
    query = 'SELECT * FROM search_engin_MeshWord WHERE `mesh` LIKE '+ sqlstr
    for p in Index.objects.filter(mesh__contains = meshWord).values('mesh'):
        p = p['mesh']
        jsonLi = []
        jsonLi = json.loads(p) 
        for word in jsonLi:
            word = re.sub('\[.+\]','',word)
            word = word.replace('(',"")
            word = word.replace(')',"")
            result.append(word)
    return list(set(result))


def search_preprocess():
    doc_lemma=[]
    for i in UploadFile.objects.all():
        doc_lemma.append(jsonDec.decode(i.lemmas))
        

    dictionary = corpora.Dictionary(token for token in doc_lemma)
    dictionary.save("./polls/pubmed.dict")

    corpus = [dictionary.doc2bow(text) for text in doc_lemma]
    corpora.MmCorpus.serialize("./polls/pubmed.mm", corpus)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=len(doc_lemma))
    corpus_lsi = lsi[corpus_tfidf]
    lsi.save('./polls/pubmed.lsi')

def startThread():
    t = threading.Thread(target=geturl)
    t.daemon = True
    t.start()
    
def geturl():
    abstr_id = 	30540103
    names=[]
    for i in UploadFile.objects.all():
        names.append(i.name)
    for i in range(3000) :
        if str(abstr_id)+'.xml' not in names:
            url = 'https://www.ncbi.nlm.nih.gov/pubmed/'+str(abstr_id)
            html = urllib.request.urlopen(url)
            parsed_html = BeautifulSoup(html,"lxml")
            text = ""
            title = ""
            if parsed_html is not None and parsed_html.body.find('div', attrs={'class':'abstract-content'}) is not None and parsed_html.body.find('h1', attrs={'class':'heading-title'}) is not None:
                text = str(parsed_html.body.find('div', attrs={'class':'abstract-content'}).find('p').text)
                text = re.sub('\n', '', text)
                text = re.sub(' +', ' ', text)
                text = text[:-1]
                text = text[1:]


                title = str(parsed_html.body.find('h1', attrs={'class':'heading-title'}).text)
                title = re.sub('\n', '', title)
                title = re.sub(' +', ' ', title)
                title = title[:-1]
                title = title[1:]

                print(str(abstr_id) + ' : ' + title)
            
                #count id
                t = UploadFile.objects.last()
                if t is not None:
                    id = t.fid + 1
                else:
                    id = 0
                
                #file processing
                file_name = str(abstr_id)+'.xml' 
                file_title = title
                file_text = text
                file_eof,_ = e_o_f(file_text)
                file_lemmas = [l.lemma_ for l in nlp(file_text) if l.is_stop is False]

                if str(abstr_id)+'.xml' not in names:
                    if file_name not in names:
                        #file data store
                        UploadFile.objects.create(
                            fid = id,
                            name = file_name,
                            title = file_title,
                            text = file_text,
                            lemmas = json.dumps(file_lemmas),
                            eof = json.dumps(file_eof),
                            )

                        #update index
                        for i in file_lemmas:
                            query_word = Index.objects.filter(lemma = i).last()
                            if query_word:
                                temp1 = query_word.position
                                temp2 = query_word.lemma
                                new_position = jsonDec.decode(query_word.position)
                                if file_name not in new_position : 
                                    Index.objects.filter(lemma = i).update(
                                        lemma = i,
                                        position = json.dumps(new_position + [file_name])
                                    )
                            else:
                                Index.objects.create(
                                    lemma = i,
                                    position = json.dumps([file_name,]),
                                    mesh = FindMeshWord(i)
                                )

        abstr_id = abstr_id+1