%define problem parameters.. done it this way for things to work.
p11 = sym('p11'); p12 = sym('p12'); p1 = [p11;p12];
m11 = sym('m11'); m12 = sym('m12'); m1 = [m11;m12];
p21 = sym('p21'); p22 = sym('p22'); p2 = [p21;p22];
m21 = sym('m21'); m22 = sym('m22'); m2 = [m21;m22];
q1 = sym('q1'); q2 = sym('q2'); q = [q1;q2];
s = sym('s');

%define hermite cubic spline == q ferguson curve
fs1 = (2*s.^3-3*s.^2+1)*p1(1) + ...
    (s.^3 - 2*s.^2 + s)*m1(1) + ...
    (-2*s.^3 + 3*s.^2)*p2(1) + ...
    (s.^3 - s.^2)*m2(1);
fs2 = (2*s.^3-3*s.^2+1)*p1(2) + ...
    (s.^3 - 2*s.^2 + s)*m1(2) + ...
    (-2*s.^3 + 3*s.^2)*p2(2) + ...
    (s.^3 - s.^2)*m2(2);

fs = [
      fs1
      fs2];

%define squared distance of point to spline to minimize... back to least
%squares. what a surprise. prehaps we can ask the solver to also find s
%that minimizes this
Dsq = sum((q - fs).^2);

%to miniize Dsq we can diff it and compare to zero
dDsq = diff(Dsq,s);

%finding closed form roots to dDsq would be impossible..

%Solution one:
%lets assume that higher terms are insignificant, and that the function Dsq
%is parabolic. As such, dDsq will be approx linear.. and we can take the
%two relevant terms and compare to zero
dDsq_coeffs = coeffs(dDsq,s);
a = dDsq_coeffs(end-1);
b = dDsq_coeffs(end);
smin1 = -b/a; %<-------------------got something?

%Solution two:
%lets assume dsq is parabolic around the minimal distance smin (which we don't
%know what it is...)
smin = sym('smin');
ddDsq = diff(dDsq,s);
Dsq_shifted = subs(Dsq,s,smin) + subs(dDsq,s,smin)*(s-smin) + subs(ddDsq,s,smin)/2*(s-smin)^2; %make taylor series
Dsq_shifted_coeffs = coeffs(Dsq_shifted,s); %Dsq_shifted is parabolic in s...
a = Dsq_shifted_coeffs(1);
b = Dsq_shifted_coeffs(2);
smin2 = solve(smin == -b/(2*a),smin); %<----------------Useless

% roots(coeffs) leads nowhere... we need to reduce the problem.
% I see two options from here:
%option (1):
    %roots(coeffs(2:6) leads to four solutions. We could check all four and
    %take the minimal. symforce's sympy can define differential for the min
    %operator.
%option (2):
    %fit a parabola to dsq around q. We have a closed formula for that


dsq_coeffs_simp = dsq_coeffs(5:7);
a = dsq_coeffs_simp(1);
b = dsq_coeffs_simp(2);
smin = -b/(2*a);


%plug in parameters to view result
p0a = 0;
m0a = 0;
p1a = 1;
m1a = 0;
q0a = 0.3;
q1a = 0.6;

%running paramter to place in s
t = linspace(0,1,100);

%find projected point
smina = double(subs(smin,[p0, m0, p1, m1, q0, q1],[p0a, m0a, p1a, m1a, q0a, q1a]));

%plot curve
figure()
f = matlabFunction(subs(fs,[p0, m0, p1, m1],[p0a, m0a, p1a, m1a]));
ft = f(t);
plot(t,ft); hold("on");
scatter(q0a,q1a);
scatter(smina, f(smina));
title('curve and point');


%plot dsq curve
figure()
t = linspace(0,1,100);
ft = subs(dsq,[p0, m0, p1 m1, q0 ,q1],[0, 0, 0, -5, 0.2,0.5]);
ft = subs(ft,s,t);
plot(t,ft); hold("on");
scatter(smin,ft(smin))
title('distance to q(s)')

