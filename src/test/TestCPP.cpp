//
// Created by DY on 17-7-7.
//

#include <iostream>
#include <vector>
#include <regex>
#include <string>
#include <cassert>

using namespace std;

const int N = 40;

int main(int argc, char* args[]) {
    //assert(argc == 3);
    //string s(args[1]);
    //string p(args[2]);
    //cout << "TEXT " << s << " PATTERN " << p << endl;
    //std::regex word_regex(p);
    //auto words_begin = 
    //    std::sregex_iterator(s.begin(), s.end(), word_regex);
    //auto words_end = std::sregex_iterator();
 
    //std::cout << "Found "
    //          << std::distance(words_begin, words_end)
    //          << " matches\n";
 
    //for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    //    std::smatch match = *i;
    //    std::string match_str = match.str();
    //        std::cout << "  " << match_str << '\n';
    //}
    int n = 10;
    string* ss = new string[n];
    ss[1] = "hello";
    ss[2] = "world";
    for (int i = 0; i < n; ++i) cout << ss[i] << endl;
    return 0;
}

