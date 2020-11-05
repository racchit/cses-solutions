#include<bits/stdc++.h>
#define ll long long
using namespace std;

int main(){

    int n;
    cin >> n;
    cout << 0 << endl;
    ll a = 6;
    ll b;
    for(ll k=3; k<=n+1; k++){
        cout<<a<<endl;
        b = a + (2*k-1)*(k-1) + (k-3)*(k-3)*(2*k-1) + (2*k-1)*(4*k-8) - (8*k-18) - 2;
        a = b;
    }

    return 0;
}

