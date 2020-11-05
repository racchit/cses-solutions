#include<bits/stdc++.h>
#define ll long long

using namespace std;

int main(){

    ll n, k;
    cin>>n;
    k=1;
    for(int i=0; i<n;i++){
        k=2*k%(1000000007);
    }

    cout << k;
    return 0;
}
