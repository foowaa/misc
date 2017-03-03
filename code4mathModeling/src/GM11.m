%purpose: grey prediction GM(1,1)
%@params: A:target
%         year: year begun
%         add: how many years do U wanna predict
%@return: G: prediction
function G=GM11(A, year, add)

B=cumsum(A); 
n=length(A);

for i=1:(n-1) 
    C(i)=(B(i)+B(i+1))/2; 
end 
D=A; 
D(1)=[]; 
D=D'; 
E=[-C;ones(1,n-1)]; 
c=inv(E*E')*E*D; 
c=c'; 
a=c(1); 
b=c(2); 
F=[]; 
F(1)=A(1); 
for i=2:(n+add) 
    F(i)=(A(1)-b/a)/exp(a*(i-1))+b/a; 
end 
G=[]; 
G(1)=A(1); 
for i=2:(n+add) 
    G(i)=F(i)-F(i-1); 
end 
t1=year; 
t2=year(1):year(end)+add; 
plot(t1,A,'o',t2,G)