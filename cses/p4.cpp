#include<bits/stdc++.h>

#define ll long long

using namespace std;

int main(){
    ll n;
    cin>>n;
    ll A[n];
    ll a=0;
    ll pos;
    for(ll i = 0; i < n; i++){
        ll x;
        cin>>x;
        A[i] = x;
        if(A[i] > a){
            a = A[i];
            pos = i;
        }

    }
    ll ans = 0;
    for(ll i = 0; i < n-1; i++){
        ll c = A[i] - A[i+1];
        if(c > 0){
            ans = ans + c;
            A[i+1] = A[i];
        }
    }
    cout<<ans;
    return 0;
}



