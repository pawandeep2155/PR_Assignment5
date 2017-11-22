data1 = [1 1 ; 1 2 ; 2 1 ;2 2 ];
data2 = [-1 1 ; -1 2 ; -2 1; -2 2];

clf;
scatter(data1(:,1),data1(:,2));
hold on;
axis([-10 10 -10 10]);
scatter(data2(:,1),data2(:,2));
hold on;

data = [data1;data2];

w2 = rand * 10;
w1 = rand * 10;
w0 = rand * 10;
w = [w2;w1;w0];

learn_rate = 0.01;

for i=1:1000
    for j=1:8
        % target
        if(w' * [data(j,:)';1] <= 0 && j<=4)
            w = w + learn_rate * [data(j,:)';1];
        end
        % non target
        if(w' * [data(j,:)';1] >0 && j>4)
            w = w - learn_rate * [data(j,:)';1];
        end
    end
end

x = -10:1:100;
y = (-w(1)*x + w(3))/w(2);
plot(x,y)


line();


