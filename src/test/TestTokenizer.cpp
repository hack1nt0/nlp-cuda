//
// Created by DY on 17-7-7.
//

#include <iostream>
#include <vector>
#include <Tokenizer.h>
#include <complex>
#include <algorithm>

using namespace std;

int main() {
    Tokenizer tokenizer;
    string s = "This is a good day!";
    vector<string> words;
    tokenizer.splitFilterSpace(words, s);
    printf("%d\n", words.size());
    for (auto i = words.begin(); i != words.end(); ++i) cout << *i << endl;
    for (auto i : words) cout << i << endl;


    return 0;
}
