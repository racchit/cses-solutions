#include<bits/stdc++.h>

#define ll long long

using namespace std;

int main(){

    ll n, k;
    cin >> n;
    k = n;
    ll sum2 = 0;
    ll sum5 = 0;
    ll i=1;
    ll term = 1;
    //highest power of 2
    while(term){
        sum2 += floor(n/pow(2,i));
        term = floor(n/pow(2,i));
        i++;
    }
    ll j=1;
    //highest power of 5
    term = 1;
    while(term){
        sum5 += floor(n/pow(5,j));
        term = floor(n/pow(5,j));
        j++;
    }
    ll ans = min(sum2,sum5);
    cout<<ans;
    return 0;
}
