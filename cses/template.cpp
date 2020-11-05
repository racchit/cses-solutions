//Tortoise Name:SINayem
/*
   ->Happy coding!
   */
/*#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>*/
#include<bits/stdc++.h>
#include<cmath>

using namespace std;
//using namespace __gnu_pbds;

#define debug printf("Debug\n");
#define en '\n'


#define ll long long
#define ull unsigned long long
#define dob double

#define pi acos(-1.0)
#define mod 1000000007
#define INF 1e18
#define _INF -1e18

#define _fastio  ios_base:: sync_with_stdio(false); cin.tie(0); cout.tie(0);
#define in_fre  freopen("in.txt", "r", stdin);
#define out_fre freopen("out.txt", "w", stdout);

struct Node{
    ll x,y;
};
struct cmp1{
    bool operator()(const Node &a,const Node &b)const{
        if(a.x==b.x) return a.y<b.y;
        else return a.x>b.x;
    }
};

//STL Algos
//template<typename T> using Pbds=tree<T,null_type,less<pair<ll,ll>>,rb_tree_tag,tree_order_statistics_node_update>;

//Number Theory
class NumberTheory{
    public:
        ll n;
        ll mindiv[200000];
        bool chk[99999995];

        ll GetMod(ll x,ll m);
        ll BigMod (ll b,ll p,ll m){if (p == 0) return 1;if (p%2 == 0){ll s = BigMod(b,p/2,m);return ((s%m)*(s%m))%m;}return ((b%m)*(BigMod(b,p-1,m)%m))%m;}
        ll ModInv (ll b,ll m){return BigMod(b,m-2,m);}
        ll gcd(ll a,ll b){if(a<0)return gcd(-a,b);if(b<0)return gcd(a,-b);return (b==0)?a:gcd(b,a%b);}
        ll lcm(ll a,ll b) {if(a<0)return lcm(-a,b);if(b<0)return lcm(a,-b);return a*(b/gcd(a,b));}
        void sieve(ll x);
        bool IsPrime(ll x);
        bool IsPrime1(ll x){return !chk[x];}
        void FactSieve(ll x);
        vector<ll> getFactorization(ll x);
};
//Geometry
class Geometry{
    public:
        struct CartesianPoint{dob x,y;};
        struct PolarPoint{dob r,ang;};
        dob SQR(dob x);
        dob DOT(dob x1,dob y1,dob x2,dob y2);
        dob CROSS(dob x1,dob y1,dob x2,dob y2);
        dob PolarR(dob x,dob y);
        dob PolarAngle(dob x,dob y);
        dob CartesianX(dob r,dob ang);
        dob CartesianY(dob r,dob ang);
        dob CartesianDis(dob x1,dob y1,dob x2,dob y2);
        dob PolarDis(dob r1,dob ang1,dob r2,dob ang2);
        dob MiddlePoint(dob x1,dob x2);
        dob InternalSection(dob x1,dob x2,dob m1,dob m2);
        dob ExternalSection(dob x1,dob x2,dob m1,dob m2);
        dob AreaOfTriangle(dob x1,dob y1,dob x2,dob y2,dob x3,dob y3);
        dob GradAngle(dob x1,dob y1,dob x2,dob y2);
        dob Grad(dob x1,dob y1,dob x2,dob y2);
        dob Line(dob m,dob c,dob x,dob y);
        dob Line(dob m,dob x1,dob y1,dob x,dob y);
        dob Line(dob x1,dob y1,dob x2,dob y2,dob x,dob y);
        dob OrthogonalLine(dob x1,dob y1,dob x2,dob y2,dob lx,dob ly,dob ox,dob oy);
        dob LineOrthogonalLineIntersecX(dob x1,dob y1,dob x2,dob y2,dob ox,dob oy);
        dob LineOrthogonalLineIntersectY(dob x1,dob y1,dob x2,dob y2,dob ox,dob oy);
        bool IsParallel(dob x1,dob y1,dob x2,dob y2);
        bool IsOrthogonal(dob x1,dob y1,dob x2,dob y2);
        dob IntersecX(dob a1,dob b1,dob c1,dob a2,dob b2,dob c2);
        dob IntersecY(dob a1,dob b1,dob c1,dob a2,dob b2,dob c2);
        dob PointToLineDis(dob a,dob b,dob c,dob x,dob y);
        dob PointToLineDis(dob x1,dob y1,dob x2,dob y2,dob x,dob y);
        dob LineToLineDis(dob a,dob b,dob c1,dob c2);
        ll isLeft(CartesianPoint a,CartesianPoint b,CartesianPoint c);

};

//Bitwise operations
class Bitwise_operation{
    public:
        ull a[65];
        ull A,B;
        bool check(ull N,ull pos){ return (bool)(N & (1LL<<pos)); }
        ull Set1(ull N,ull pos){ return N=N | (1LL<<pos); }
        ull Set0(ull N,ull pos){ return N=N & ~(1LL<<pos); }
        ull flip(ull N,ull pos){ return N=N ^ (1LL<<pos); }
        bool isPowerOf2(ull n){return !(n&(n-1))?true:false;}
        void swap(){ A = A ^ B;B = A ^ B;A = A ^ B;}
        ull FindLargestPowerof2(ull N){N=N|N>>1;N=N|N>>2;N=N|N>>4;N=N|N>>8;N=N|N>>16;N=N|N>>32; return(N+1)>>1;}
        ull GetRightmost1(ull x){return (x & (-x));}
        ull GetRightmost1spos(ull x){return __builtin_ffs(x)-1;}
        ll CountNumberof1s(ull x){ll cnt=0;while( x ){x = x&(x-1);cnt++;}return cnt;}
        ll CountLeadingZeros(ull x){return __builtin_clz(x);}
        ll CountTrailingZeros(ull x){return __builtin_ctz(x);}
        void possibleSubsets(ll N);

};

//Binary Search
class Binry_Search{
    public:
        ll ar[200005];
        ll Manual_upper(ll l,ll r,ll val);
        ll Manual_lower(ll l,ll r,ll val);
};

//Graph Theory
class GraphTheory_BFS_DFS{
    public:
        ll nodes,edges;
        vector<ll>graph[200005];ll vis[200005];ll cnt[200005];
        void CLR(ll n);
        bool isLeaf(ll node);
        void Connect(ll u,ll v);
        void bfs(ll src);
        void dfs(ll src);
        void CountSubtreeSize(ll cur);
        ll dirx[5]={1,-1,0,0};
        ll diry[5]={0,0,1,-1};
        void GirdBfs(ll i,ll j);
        void GridDfs(ll i,ll j);
};
//Data Structure
class DataStructure_DSU{
    public:
        ll parent[200005];ll Size[200005];ll Rank[200005];
        void CLR(ll n);
        void make_set(ll v);
        ll find_set(ll v);
        void union_sets(ll a, ll b);
        ll Path_Compressed_find_set(ll v);
        void union_sets_considering_size(ll a, ll b);
        void union_sets_considering_rank(ll a, ll b);
        void union_sets_considering_size_without_path_compression(ll a, ll b);
};

class DataStructure_Trie{
    public:
        bool f;
        ll tot;
        ll tri[400005][30],emark[400005],cnt[400005],depth[400005];
        string s;
        void CLR(ll n);
        void build();
        ll Find();
};

class DataStructure_Segment_Tree{
    public:
        ll n,tot;
        ll el[100005],segtree[400005];
        void CLR();
        void build(ll node,ll b,ll e);
        void up(ll node,ll b,ll e,ll p);
        void up1(ll node,ll b,ll e,ll l,ll r);
        void query(ll node,ll b,ll e,ll p);
        void query1(ll node,ll b,ll e,ll l,ll r);
};

class DataStructure_SQRT_DCOM{
    public:

};
long long binpow(long long a, long long b, long long m) {
    a %= m;
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
}
vector<ll>v;
deque<ll>dq;
stack<ll>stk;
map<ll,ll>mp;
unordered_map<ll,ll>ump;
set<ll>st;
multiset<ll>mst;
unordered_set<ll>ust;
unordered_multiset<ll>umst;
priority_queue<ll,deque<ll>,greater<ll>>pq;
set<Node,cmp1>st1;
map<Node,ll,cmp1>mp1;
priority_queue<Node,deque<Node>,cmp1>pq1;
//Pbds<pair<ll,ll>>ost;
double h,c,t;

double chk(ll x){
    ll m1=x/2;
    ll m2=x/2;
    if(x%2) m1++;
    double ans=(m1*h+m2*c)/(x*1.00);
    return ans;
}

int main(){
    _fastio
        ll test,n;
    cin>>n;
    ll ar[n];
    ll c=0;
    for(ll i=0;i<n;i++)cin>>ar[i];
    for(ll i=0;i<n-1;i++){
        if(ar[i+1]-ar[i]<0){
            c+=ar[i]-ar[i+1];
            ar[i+1]+=ar[i]-ar[i+1];
        }

    }
    cout<<c;
    return 0;
}
/*7
  25 -13 17 -23 12 12 9*/

//NumberTheory
void NumberTheory::sieve(ll x){
    for(ll i=2;i*i<=x;){
        if(chk[i]==0){
            for(ll j=i*i;j<=x;j+=i){
                chk[j]=1;
            }
        }
        if(i==2) i++;
        else i+=2;
    }
}
bool NumberTheory::IsPrime(ll x){
    bool f=true;
    for(ll i=2;i*i<=x;i++){
        if(!x%i){
            f=false;break;
        }
    }
}
ll GetMod(ll x,ll m){
    if(x>=0) return x%m;
    else{
        ll y=abs(x)/m;x+=(y*m+m+m);
        return x%m;
    }
}
void NumberTheory::FactSieve(ll x){
    mindiv[1] = 1;
    for (ll i=2; i<x; i++) mindiv[i] = i;
    for (ll i=4; i<x; i+=2) mindiv[i] = 2;
    for (ll i=3; i*i<x; i++){
        if (mindiv[i] == i){
            for (ll j=i*i; j<x; j+=i)
                if (mindiv[j]==j)
                    mindiv[j] = i;
        }
    }
}
vector<ll> NumberTheory::getFactorization(ll x){
    vector<ll> ret;
    while (x != 1){
        ret.emplace_back(mindiv[x]);
        x = x / mindiv[x];
    }
    return ret;
}

//Bitwise operations
void Bitwise_operation::possibleSubsets(ll N){
    for(ll i = 0;i <(1 << N); ++i){
        for(ll j = 0;j < N;++j)
            if(check(i,j))
                cout << a[j]<<' ';
        cout << endl;
    }
}

//BinarySearch
ll Binary_Search::Manual_upper(ll l,ll r,ll val){
    ll x=val;
    ll pos=0;
    while(1){
        ll mid=(l+r)>>1;
        if(r-l<=1){
            if(ar[r]==x) pos=r;
            else if(ar[l]==x) pos=l;
            else pos=l;
            break;
        }
        else if(x>=ar[mid]) l=mid;
        else r=mid;
    }
    return pos;
}
ll Binary_Search::Manual_lower(ll l,ll r,ll val){
    ll x=val;
    ll pos=0;
    while(1){
        ll mid=(l+r)>>1;
        if(r-l<=1){
            if(ar[l]==x) pos=l;
            else if(ar[r]==x) pos=r;
            else pos=l;
            break;
        }
        else if(x<=ar[mid]) r=mid;
        else l=mid;
    }
    return pos;
}

//Graph Theory
void GraphTheory_BFS_DFS::CLR(ll n){
    for(ll i=0;i<=n+1;i++){
        graph[i].clear();
        vis[i]=0;
    }
    nodes=n;
}
bool GraphTheory_BFS_DFS::isLeaf(ll node){
    if(graph[node].size()==1) return true;
    else return false;
}

void GraphTheory_BFS_DFS::Connect(ll u,ll v){
    graph[u].push_back(v);
    graph[v].push_back(u);
}
void GraphTheory_BFS_DFS::bfs(ll src){
    vis[src]=1;
    deque<ll>dq;
    dq.push_back(src);
    while(!dq.empty()){
        src=dq.front();
        dq.pop_front();
        for(ll i=0;i<graph[src].size();i++){
            ll adj=graph[src][i];
            if(vis[adj]==0){
                vis[adj]=1;
                dq.push_back(adj);
            }
        }
    }
}
void GraphTheory_BFS_DFS::dfs(ll src){
    vis[src]=1;
    for(ll i=0;i<graph[src].size();i++){
        ll adj=graph[src][i];
        if(!vis[adj]){
            dfs(adj);
        }
    }
}
void GraphTheory_BFS_DFS::CountSubtreeSize(ll cur){
    vis[cur]=1;
    cnt[cur]++;
    for(ll i=0;i<graph[cur].size();i++){
        ll adj=graph[cur][i];
        if(!vis[adj]){
            CountSubtreeSize(adj);
            cnt[cur]+=cnt[adj];
        }
    }
}



//Data Structure
/*
   Using DSU.
   1.we can find number of connected components of a graph
   2.we can check either two vertex are in same component or different
   3.we can find how many vertex is in each component
   4.we can find minimum spanning tree
   5.using find_parent function without path compression we can find any ancestor of a vertex. that means we can find lowest common ancestor
   6.Also, it's worth mentioning that DSU with union by size / rank, but without path compression works in O(logn) time per query.
   7.
   */
void DataStructure_DSU::CLR(ll n){
    for(ll i=0;i<=n+1;i++){
        parent[i]=i;
        Rank[i]=0;
        Size[i]=1;
    }
}
void DataStructure_DSU::make_set(ll v) {
    parent[v] = v;
}
ll DataStructure_DSU::find_set(ll v) {
    if (v == parent[v])
        return v;
    return find_set(parent[v]);
}
void DataStructure_DSU::union_sets(ll a, ll b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b)
        parent[b] = a;
}
ll DataStructure_DSU::Path_Compressed_find_set(ll v) {
    if (v == parent[v])
        return v;
    return parent[v]=Path_Compressed_find_set(parent[v]);
}
void DataStructure_DSU::union_sets_considering_size(ll a, ll b) {
    a = Path_Compressed_find_set(a);
    b = Path_Compressed_find_set(b);
    if (a != b) {
        if (Size[a] < Size[b])
            swap(a, b);
        parent[b] = a;
        Size[a] += Size[b];
    }
}
void DataStructure_DSU::union_sets_considering_rank(ll a, ll b){
    a = Path_Compressed_find_set(a);
    b = Path_Compressed_find_set(b);
    if (a != b) {
        if (Rank[a] < Rank[b])
            swap(a, b);
        parent[b] = a;
        if (Rank[a] == Rank[b])
            Rank[a]++;
    }
}
void DataStructure_DSU::union_sets_considering_size_without_path_compression(ll a, ll b){
    //log(n)
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (Size[a] < Size[b])
            swap(a, b);
        parent[b] = a;
        Size[a] += Size[b];
    }
}


/*
 *We can check whether an string is a prefix/suffix of any set of strings OR any string from a set of strings is a prefix/suffix of that string or not.
 *We can find how many times a string appears as a prefix/suffix in a set of strings.
 *We can find longest/shortest common prefix/suffix
 *We can find longest common substring
 */

void DataStructure_Trie::CLR(ll n){
    for(ll i=0;i<=n+1;i++){
        memset(tri[i],0,sizeof(tri[i]));
        emark[i]=0;
        cnt[i]=0;
    }
    s.clear();
    tot=0;
}

void DataStructure_Trie::build(){
    ll now=0;
    ll len=s.length();
    for(ll i=0;i<len;i++){
        ll x=s[i]-'0';
        if(tri[now][x]==0){
            tot++;
            tri[now][x]=tot;
            depth[tot]=depth[now]+1;
            now=tot;
        }
        else{
            now=tri[now][x];
        }
        cnt[now]++;
        if(i==len-1) emark[now]++;
    }
}

ll DataStructure_Trie::Find(){
    ll ans=0;
    ll now=0;
    ll len=s.length();
    for(ll i=0;i<len;i++){
        ll x=s[i]-'0';
        if(tri[now][x]==0){
            return 0;
        }
        else{
            now=tri[now][x];
        }
        if(i==len-1){
            ans=cnt[now];
        }
    }
    return ans;
}
/*
   A lot of works we can do . :p
   */
void DataStructure_Segment_Tree::CLR(){
    memset(segtree,0,sizeof(segtree));
}
void DataStructure_Segment_Tree::build(ll node,ll b,ll e){
    if(b==e){
        segtree[node]=el[b];
        return;
    }
    ll mid=(b+e)/2;
    ll left=2*node;
    ll right=left+1;
    build(left,b,mid);
    build(right,mid+1,e);
    segtree[node]=segtree[left]+segtree[right];
    return;

}
void DataStructure_Segment_Tree::up(ll node,ll b,ll e,ll p){
    if(b==e){
        if(b==p)
            segtree[node]=el[b];
        return;
    }
    if(b>p||e<p) return;

    ll mid=(b+e)/2;
    ll left=2*node;
    ll right=left+1;
    up(left,b,mid,p);
    up(right,mid+1,e,p);
    segtree[node]=segtree[left]+segtree[right];
    return;
}
void DataStructure_Segment_Tree::up1(ll node,ll b,ll e,ll l,ll r){
    if(b==e){
        if(b>=l&&e<=r)
            segtree[node]++;
        return;
    }
    if(b>r||e<l) return;

    if(b>=l&&e<=r){
        segtree[node]++;
        return;
    }

    ll mid=(b+e)/2;
    ll left=2*node;
    ll right=left+1;
    up1(left,b,mid,l,r);
    up1(right,mid+1,e,l,r);
}
void DataStructure_Segment_Tree::query(ll node,ll b,ll e,ll p){
    if(b==e){
        if(b==p)
            tot+=segtree[node];
        return;
    }
    if(b>p||e<p) return;

    tot+=segtree[node];

    ll mid=(b+e)/2;
    ll left=2*node;
    ll right=left+1;
    query(left,b,mid,p);
    query(right,mid+1,e,p);
    return;
}
//void query(ll node,ll b,ll e,ll l,ll r);

dob Geometry::SQR(dob x){return x*x;}
dob Geometry::DOT(dob x1,dob y1,dob x2,dob y2){ return x1*x2+y1*y2;}
dob Geometry::CROSS(dob x1,dob y1,dob x2,dob y2){ return x1*y2-x2*y1;}
dob Geometry::PolarR(dob x,dob y){ return sqrt(SQR(x)+SQR(y));}
dob Geometry::PolarAngle(dob x,dob y){ return atan2(dob(y),dob(x));}
dob Geometry::CartesianX(dob r,dob ang){ return r*cos(dob(ang));}
dob Geometry::CartesianY(dob r,dob ang){ return r*sin(dob(ang));}
dob Geometry::CartesianDis(dob x1,dob y1,dob x2,dob y2){return sqrt( SQR(x1-x2) + SQR(y1-y2) );}
dob Geometry::PolarDis(dob r1,dob ang1,dob r2,dob ang2){return sqrt(SQR(r1)+SQR(r2)-2.00*r1*r2*cos(ang1-ang2));}
dob Geometry::MiddlePoint(dob x1,dob x2){return (x1+x2)/2;}
dob Geometry::InternalSection(dob x1,dob x2,dob m1,dob m2){ return (m1*x2+m2*x1)/(m1+m2);}
dob Geometry::ExternalSection(dob x1,dob x2,dob m1,dob m2){ return (m1*x2-m2*x1)/(m1-m2);}
dob Geometry::AreaOfTriangle(dob x1,dob y1,dob x2,dob y2,dob x3,dob y3){ return (x1*y2+x2*y3+x3*y1)-(y1*x2+y2*x3+y3*x1);}
dob Geometry::GradAngle(dob x1,dob y1,dob x2,dob y2){ return atan2( dob(y1-y2) , dob(x1-x2));}
dob Geometry::Grad(dob x1,dob y1,dob x2,dob y2){ return (x1-x2)==0.00?1.00*1e18:( dob(y1-y2) / dob(x1-x2));}
dob Geometry::Line(dob m,dob c,dob x,dob y){ return y-m*x-c;}
dob Geometry::Line(dob m,dob x1,dob y1,dob x,dob y){ return (y-y1)-m*(x-x1);}
dob Geometry::Line(dob x1,dob y1,dob x2,dob y2,dob x,dob y){ return (y2-y1)*x-(x2-x1)*y+(x1*y2-y1*x2);}
dob Geometry::OrthogonalLine(dob x1,dob y1,dob x2,dob y2,dob lx,dob ly,dob ox,dob oy){ return -lx*(x2-x1)-ly*(y2-y1)+(y2-y1)*ox+(x2-x1)*oy;}
dob Geometry::LineOrthogonalLineIntersecX(dob x1,dob y1,dob x2,dob y2,dob ox,dob oy){ return ((x1-x2)*((y2-y1)*ox+(x2-x1)*oy)-(y1-y2)*(x1*y2-x2*y1))/((y2-y1)*(y1-y2)-SQR(x1-x2));}
dob Geometry::LineOrthogonalLineIntersectY(dob x1,dob y1,dob x2,dob y2,dob ox,dob oy){ return ((x1-x2)*(x1*y2-x2*y1)-(y2-y1)*((y2-y1)*ox+(x2-x1)*oy))/((y2-y1)*(y1-y2)-SQR(x1-x2));}
bool Geometry::IsParallel(dob x1,dob y1,dob x2,dob y2){ (CROSS(x1,y1,x2,y2)==0.00)?true:false;}
bool Geometry::IsOrthogonal(dob x1,dob y1,dob x2,dob y2){ (DOT(x1,y1,x2,y2)==0.00)?true:false;}
dob Geometry::IntersecX(dob a1,dob b1,dob c1,dob a2,dob b2,dob c2){ return (b1*c2-b2*c1)/(a1*b2-a2*b1);}
dob Geometry::IntersecY(dob a1,dob b1,dob c1,dob a2,dob b2,dob c2){ return (b1*c2-b2*c1)/(a1*b2-a2*b1);}
dob Geometry::PointToLineDis(dob a,dob b,dob c,dob x,dob y){ return (a*x+b*y+c)/sqrt(SQR(a)+SQR(b));}
dob Geometry::PointToLineDis(dob x1,dob y1,dob x2,dob y2,dob x,dob y){ return PointToLineDis(y2-y1,x1-x2,x1*y2-x2*y1,x,y);}
dob Geometry::LineToLineDis(dob a,dob b,dob c1,dob c2){ return abs(c1-c2)/sqrt(SQR(a)+SQR(b));}
ll Geometry::isLeft(CartesianPoint a,CartesianPoint b,CartesianPoint c) { return (a.x-b.x)*(b.y-c.y)-(b.x-c.x)*(a.y-b.y);}

/*
   The weapons in my hand:
 **STL
 *Number theory
 **Geometry

 *sqrt decomposition
 *mo's algorithm
 **segment tree
 **Disjoint set union (DSU)
 *trie/ prefix tree
 *merge sort tree
 *tree dp
 *pbds to delete elements in logn time

 **bfs
 **dfs
 *2d grid a bfs
 *2d grid a dfs
 *lca
 **heavy light decomposition/sack/dsu on tree

 *classical dp
 *bitmask dp
 *digit dp

 *hashing
 *matrix exponenciation
 *cumulative sum
 *sliding window technique.
 *two pointers
 *Greedy
 *sparse table
 *Binary Search & Binary search on answer

*/

