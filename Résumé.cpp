#include <iostream>
#include <string>
// cstring
#include <math>
#include <sstream>
#include <vector>
#include <regex>
#include <fstream>
#include <ctime>
#include <list>
#include <set>
#include <map>
#include <stack>
#include <queue>
#include <algorithm> // find, sort, copy, fill
#include <cstdlib> // rand
using namespace std;
void myFunc(int x){
    cout << x;
}
// option de passer par réf dans une fonction
void addStr(string &orig) {
    orig+="MOFIDIÉ";
}
// function overloading: un seul nom utilisé plusieurs fois pour de diff types
// scope, recursion, classe, méthodes peuvent être définie à l'int ou à l'ext
// public: accessible à l'ext de la classe
// private: non accessible/ lisible à l'ext de la classe
// protected: private sauf qu'il est accessible dans une classe héritée
// encapsulation: private property but get set method public to protect sensitive info
// in Python, _ for protected and __ for private
// polymorphism
class MyClass {
    private:
        float salaire;
    public:
        int id;
        string title;
        int age(int ag);
        void sayHi() {
            cout << "Hola amigo";
        }
        void setSalaire(float x){
            salaire=x;
        }
        float getSalaire() {
            return salaire;
        }
        // constructor
        MyClass(int x,string y) {
            id=x;
            title=y;
        }
    
};
// classe dérivée
// il est possible de dériver des classes avec une hiérarchie et dériver une classe de plusieurs classes
class MyBetterClass:public MyClass {//, public MyOtherClass
    public:
        string grade="A";
};

// définir une méthode à l'ext de la classe
int MyClass::age(int ag) {
    return ag;
}

int main() {

    // cin, cout, types de variable, const, +-*/%^, if
    const float pi=3.1415926;
    //cout << "Try programiz.pro"<<endl;
    string name="ahmed";
    double test=2.3135336465737;
    int x=2,x2=4,x3=90;
    int x5;
    int x4=x5=(x==x2)?900:890; // raccourci pour if
    float b=5.3;
    char v='a';
    std::string str_random(1,v); // convertir un char en string
    char v_ascii=67;// il se peut déclarer un char à l'aide d'un code ASCII
    cout << v_ascii<<endl; 
    bool success=false;
    
    x<<=1; // déplacer les chiffres en binaire d'une place
    cout << x+x2+x3<<endl;
    if (&x!=NULL) { // il faut comparer le pointeur avec NULL 
        cout << "y is equal to "<<x<<endl;
    } else if (1==1) {
        x++;
    } else {
        cout << success << endl;
    }
    int entry;
    //cout << "iktb:";
    //cin >> entry; problème: cin considère un espace comme un char de fin, il faudrait
    //donc utiliser getline(cin,fullName) pour saisir une ligne
    //cout << "enta ktbt " << entry;
    int sx=30;
    do {
        cout << &sx<<endl;
    } while (1<0);
    
    int* ptr=&x;
    int chapiter=1;
    switch (chapiter) {
        case 1:
            cout << *ptr<<endl; 
        default:
            cout << ptr<<endl;
    }
    int* ptr2 = new int(20);
    cout << ptr2<<endl;
    
    
    // string, .empty(), .substr(7,5), .find("s"), .rfind("str"), .replace(7,5,"sth")
    //.insert(5,"good"), .erase(3,4), str1.compare(str2)
    string s1="Nihao";
    string s2="Bonjour";// Pour accéder aux chars: s1.at(0),s2[0]
    cout << (s1.append(s2)).length(); //.size() marche aussi
    string ligne;
    //getline(cin,ligne); pour capturer une ligne avec des espaces
    cout<<ligne<<endl;
    char greeting[]="Hello"; // fonctionne comme un string mais du type char[], nommé c-style string
    // pour remplacer un substring dans un string, il faut utiliser find + replace
    string toBeReplaced="Niwep wake Nilo looooNi hahaNi";
    string sub="Ni";
    string sub_after="Mm";
    int sub_size=sub.size();
    int counter=0;
    while (toBeReplaced.find(sub) != std::string::npos) {
        cout << counter;
        toBeReplaced=toBeReplaced.replace(toBeReplaced.find(sub),sub_size,sub_after);
    }
    cout << "Après de remplacer le substring dans tous les endroits "<<toBeReplaced<<endl;
    // pour split
    string toBeSplit=toBeReplaced;
    char divider=' ';
    vector<string> split_words;
    string split_word;
    stringstream string_stream(toBeSplit);
    while (getline(string_stream,split_word,divider)){
        split_words.push_back(split_word);
    }
    // loop pour un vecteur
    for (const auto& str: split_words) {
        cout << str << " ";
    }
    cout << endl;
    
    // laternativement, on peut utiliser regex
    regex re(" ");
    sregex_token_iterator iter(toBeSplit.begin(),toBeSplit.end(),re,-1);
    sregex_token_iterator end;
    vector<string> split_words2(iter,end);
    for (const auto& str2: split_words2) {
        cout<< str2 << ",";
    }
    
    // math sqrt, round, log, max, min, abs, acos, acosh, ceil, cbrt, exp, exp2, floor
    // log2, pow(x,y)
    // while, do while, for, for :, array, break, continue
    for (int c=0; c<=10;c+=5){// ou ++c
        cout<<c;
    }
    // sizeof(): pas nombre d'éléments mais taille en bytes, int=4 bytes
    // possible d'avoir une matrice de plus d'une dim => string letters[2][4];
    int nums[3]={1,2,4}; // pas obligatoire de préciser la taille, mais la taille est nécessaire pour réserver les espaces de mémoires, taille n'est pas dynamique, contrairement aux vecteurs
    cout << sizeof(nums)/sizeof(nums[0])<<endl; // nombre d'éléments
    int res=0;
    for (int num : nums) {
        res+=num;
    }
    cout<<res;
    
    
    // struct, enum, ref, pointer
    struct {
        int id;
        string title;
    } book1,book2; // book1.title
    enum Level {
        LOW,//par défault =0; mais on peut spécifier la valeur LOW=90
        HIGH
    };
    enum Level myLevel=HIGH;
    enum Level myLevel2=myLevel;
    myLevel=LOW;
    cout << myLevel2 << endl;
    // cas où & n'est pas nécessaire pour affecter un pointeur
    // 1. création d'une variable ou d'une instance de struct; 2. avec un array, un vecteur 3. une fonction
    string* ptrString=new string("broki");
    struct Book {
        int id;
        string nom;
    };
    Book* ptrBook=new Book();
    int* ptrArrayInt =nums; // équiv à &nums[0], l'addr du premier élément
    cout << *(ptrArrayInt+1) << endl; // pour accéder au deuxième élément
    char* ptrVec=split_words[0].data();
    cout << (ptrVec) << endl;
    delete ptrString;
    void (*ptrFunc)(int)=myFunc;
    ptrFunc(45);
    string origString="FERNANDO";
    addStr(origString);
    cout << origString;
    
    // obj
    MyClass myObj(2,"Holi tio ");
    myObj.id=3;
    //myObj.title="ni hao ma";
    cout<<myObj.age(1000)<<endl;
    myObj.setSalaire(100.0);
    cout<<myObj.getSalaire();
    
    // files ofstream: créer/ écrire, ifstream: lire, fstream
    ofstream MyFile("filename.txt");
    MyFile << "Nihao funnnnny";
    MyFile.close();
    ifstream MyReadFile("filename.txt");
    string myText;
    while (getline (MyReadFile,myText)) {
        cout << myText;
    }
    MyReadFile.close();
    
    // exceptions
    try {
        throw 505;  
    } catch (int e) {
        cout << "Code err:"<<e<<endl;   
    }
    
    // dates - obtenir le temps actuel
    time_t timestamp;
    time(&timestamp); // *localtime (retourne un pointeur donc besoin d'enlever la réf), *gmtime
    cout << ctime(&timestamp)<<endl;//Fri Jan 10 23:39:35 2025
    // alternativement
    time_t ts=time(NULL);
    cout << ts<<endl; //1736552375
    
    // deux types: time_t ou struct tm
    struct tm datetime;
    time_t timestamp3;
    datetime.tm_year=2024-1900;
    datetime.tm_mon=12-1;
    datetime.tm_mday=17;
    datetime.tm_hour=12;
    datetime.tm_min=24;
    datetime.tm_sec=54;
    datetime.tm_isdst=-1;
    timestamp3=mktime(&datetime);
    cout << ctime(&timestamp3);
    cout << asctime(&datetime);// mais cette fonction ne corrige pas de dates incorrectes
    // weekday
    string weekdays[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    cout << "Weekday:"<<weekdays[datetime.tm_wday];
    
    // convertir les dates en strings
    time_t timestamp4=time(NULL);
    struct tm datetime4=*localtime(&timestamp4);
    char output[50];
    // a:Fri, b:Dec, B:December, d: mois[00], e: mois [ 0], H: H24, I: H12, M: min
    // p: AMPM, S: sec, y-: yy, Y: yyyy
    strftime(output,50,"%B %e, %Y",&datetime4);
    cout << output << "\n";
    
    // calcul
    clock_t before=clock();
    cout<<difftime(ts,timestamp3);// retourne la diff en sec
    clock_t duration=clock()-before;
    cout << "Duration:"<<(float)duration/CLOCKS_PER_SEC<<" secs";
    
    // structures de données: vecteur, liste, stack, queue, deque, set, map
    // STL: standard template library -> container, iterator, algorithm
    // vector methods: front() back() at() push_back("x") pop_back() size() empty() erase(int pos) erase(x,x+y+1) begin()
    // list: front, back, push_front, push_back, pop_front, pop_back, empty, size
    list<string> cars1 = {"Volvo", "BMW", "Ford", "Mazda"};
    // stack: LIFO push top pop size empty
    // queue: FIFO 
    queue<string> cars2;
    stack<string> cars3;
    cars3.push("Volve");
    cars3.push("Mezda");
    // deque: double-ended queque; from both ends; random access, front, end, at, push_front, push_back, pop_front, pop_back,size, empty
    deque<string> cars4 = {"Volvo", "BMW", "Ford", "Mazda"};
    // set: uniq element, sorted automatically in asc order, duplicates are ignored, no random access, adding and removing are allowed
    // insert("sth"), erase("sth"), clear(), size, empty
    set<string> cars5;
    // if desc is wanted
    set<int,greater<int>> numberSet={1,6,7,8,3};
    
    // map par défault trié à l'ordre ascendant de ses clés, clés doivent être uniques
    // pour renverser l'ordre: map<string,int,greater<string>>
    map<string,int> people={{"John",32},{"Mike",56}};
    cout << "John is "<<people["John"]; // at("John"), modifiable
    people["Anja"]=100; // ajouter, ou avec insert({xx,yy})
    // insersion avec la même clef => seulement la première valeur est sauvegardée
    people.erase("John");//supprimer, clear(), empty() checker si vide
    for (auto person:people) {
        cout << person.first<<" is "<<person.second<<endl;
    }
    
    vector<string>::iterator it;
    // iterator: pointe à vecteurs, sets, etc.; end() pointe à la position après le dern élément; pour renverser l'ordre, utiliser plutôt rbegin et rend; il marche avec liste, set, deque et map aussi
    for (it=split_words.begin();it!=split_words.end();++it) {
        cout << *it << endl;
    }
    for (auto it2=people.rbegin();it2!=people.rend();++it2){
        cout << it2 -> first << " is " << it2 -> second <<endl;
    }
    
    // trier, chercher, upper_bound(début,fin,seuil) - premier élément plus grand que le seuil, min_element, max_element, copier
    sort(split_words.begin()+0,split_words.end());// rbegin et rend pour renverser l'ordre
    auto it_word=find(split_words.begin(),split_words.end(),"Mmlo");
    
    
    vector<int> copiedNum(6); // initialiser un vecteur sans l'affecter
    vector<int> origNum={1, 7, 3, 5, 9, 2};
    // remplir avec une seule valeur: fill(copiedNum.begin(),copiedNum.end(),46);
    copy(origNum.begin(),origNum.end(),copiedNum.begin());
    
    // rand de 0 à 100
    int randomNum = rand() % 101;
    
    
    
    
    
    
    
    
    
    
    
    return 0;
}
