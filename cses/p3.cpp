#include<bits/stdc++.h>
#define ll long long

using namespace std;

int main(){

    string s;
    cin >> s;
    int n = s.length();
    int ans = 0;
    int c =0;
    for(int i = 0; i < n; i++){
        if(i==0){
            c++;
        }
        else if(s[i] == s[i-1]){
            c++;
        }
        else{
            c = 1;
        }
        if(c > ans){
            ans = c;
        }
    }
    cout << ans;


    return 0;
}
