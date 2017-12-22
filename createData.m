function[o] = createData(A,B)
n = size(A);
for i=1:n
   if (A(i) == B(i))
       o(i)=0;
   else
       o(i)=1;
   end
end