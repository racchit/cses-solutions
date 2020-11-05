#include<bits/stdc++.h>
#define ll long long

using namespace std;

int main(){

    ll n;
    cin >> n;
    if((n*(n+1)/2)%2!=0){
        cout<<"NO";
    }
    if(n%4 == 0){
        cout<<"YES"<<endl;
        cout<<n/4<<endl;
        for(int i = 0; i<n/4; i++){
            cout<<2i+1;
            cout<<n-2*i;
        }
    }
    return 0;
}

