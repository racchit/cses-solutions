#include<bits/stdc++.h>

#define ll long long

using namespace std;

int main(){
    ll t;
    cin>>t;
    while(t--){
        ll a,b;
        cin>>a>>b;
        ll x = 2*a-b;
        ll y = 2*b-a;
        if((x%3==0)&&(y%3==0)&&(x>=0)&&(y>=0)){
            cout<<"YES"<<endl;
        }
        else{
            cout<<"NO"<<endl;
        }
    }
    return 0;
}
