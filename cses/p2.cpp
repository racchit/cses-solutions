#include<bits/stdc++.h>

using namespace std;

int main(){
    long long sum = 0;
    long long n;
    cin >> n;
    for(long long i = 0; i < n-1; i++){
        long long a;
        cin >> a;
        sum += a;
    }
    long long c = (n*(n+1)/2) - sum;
    cout << c;
    return 0;
}
