function EKFLocalisation_known_matchings

% ALGORITHM PARAMETERS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nSteps = 4000;                                                   % number of simulation steps
T = .1;                                                                 % Sample time.
DrawEveryNFrames = 100;                                % how often shall we draw?
nSigmaRobot=3.03;                                                 % sigma boundary used to plot the robot ellipse

% Sensor model
SensorSettings.FieldOfView = 45;                      % in degrees (-degrees < 0 < +degrees)
SensorSettings.Range = 100;                             % in meters
SensorSettings.Rng_noise = 0.5;                        % in meters (std deviation)
SensorSettings.Brg_noise = 3;                            % in degrees (std deviation)
SensorSettings.Faultingk0=2000;                       % start simulating a sensor fault at iteration Faultingk0
SensorSettings.FaultGap=500;                           % fault lasts for  FaultGap iterations
StdOdometryNoise=[0.01; 0.01; deg2rad(1)];    % in metres & degrees

% Map definition
numberOfFeatures=100;
MapSize=100;                                                                       % The map is an square. This is the side size in meters.
Map = MapSize*rand(2,numberOfFeatures)-MapSize/2;      % Create a random map of poinf features

% Acceleration noise to change the robot behaviour
StdAccNoiseX=0.0;                                                            % X axis acceleration noise. No sideslip.
StdAccNoiseY= 0.001;                                                         % Y axis acceleration noise
StdAccNoiseYaw = deg2rad(0.25);                                      % anglular acceleration noise
StdAccNoise=[StdAccNoiseX; StdAccNoiseY; StdAccNoiseYaw];

%initial conditions:
initialPositionVelocity= [ randn(3,1)*MapSize/8; 0; 0.2; 0;];
xVehicleTrue =initialPositionVelocity;

% Initialization of store arrays for data to be plotted
PStore = zeros(3,nSteps);
XStore= zeros(3,nSteps);
XErrStore = zeros(3,nSteps);
FeatureVisibleStore=zeros(nSteps);
InnovStore = zeros(2,nSteps);
SStore= zeros(2,nSteps);

% LOCALIZATION STARTS HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xVehicleTrueLast = xVehicleTrue;
x_B_kIk_1 = xVehicleTrue(1:3);
P_B_kIk_1 = diag([1,1,deg2rad(1)]).^2;

initial_graphics(Map, x_B_kIk_1,P_B_kIk_1,nSigmaRobot);

for k = 2:nSteps

    %do world iteration
    xVehicleTrueLast=xVehicleTrue;
    xVehicleTrue=SimulateWorld(T, StdAccNoise, xVehicleTrueLast);
    %pause

    %figure out control
    [uk, Qk]=get_odometry(StdOdometryNoise,xVehicleTrue, xVehicleTrueLast);

    %do prediction
    [x_B_kIk_1, P_B_kIk_1] = move_vehicle(x_B_kIk_1, P_B_kIk_1, uk, Qk);

    %observe a random feature meas
    [zk, Rk, cFeature, mFeature] = get_measurements(Map, k, xVehicleTrue, SensorSettings);
%     zk=[];

    if (~isempty(zk))
        [x_B_k, P_B_k, Vk, S] = update_position(x_B_kIk_1, P_B_kIk_1, zk, Rk, cFeature);
        InnovStore(:,k) = Vk;
        SStore(:,k) = sqrt(diag(S));
        FeatureVisibleStore(k)=1;
    else
        x_B_k = x_B_kIk_1;
        P_B_k = P_B_kIk_1;
        FeatureVisibleStore(k)=0;
    end

    draw_map(Map, x_B_k, P_B_k, nSigmaRobot, zk, mFeature, DrawEveryNFrames, k)

    %store results:
    PStore(:,k) = sqrt(diag(P_B_k));
    XStore(:,k) = x_B_k;
    XErrStore(:,k) = xVehicleTrue(1:3) - x_B_k;

    %test Theta for better ploting
    XErrStore(3,k) = AngleWrap(XErrStore(3,k));

    %prepare next iteration
    xVehicleTrueLast = xVehicleTrue;
    x_B_kIk_1 = x_B_k;
    P_B_kIk_1 = P_B_k;
end

DoGraphs(InnovStore,PStore,SStore,XErrStore,FeatureVisibleStore);
% LOCALIZATION ENDS HERE %%%%%%%%%%%%%%%%%%%%%%%%%%

% SIMULATION OF THE ROBOT MOTION, THE ODOMETRY AND THE MEASUREMENTS %%%%%
% This function simulates how the robot moves through the environment
function xVehicleTrue=SimulateWorld(T,StdAccNoise, xVehicleTrueLast)
yaw=xVehicleTrueLast(3);
ERG=[cos(yaw) -sin(yaw) 0
    sin(yaw) cos(yaw) 0
    0 0 1];
noise=randn(3,1) .*StdAccNoise;
xVehicleTrue=[eye(3,3) ERG*T; zeros(3,3) eye(3,3)]*xVehicleTrueLast+[ERG zeros(3,3); zeros(3,3) eye(3,3)]*[eye(3,3)*(T^2)/2 ; eye(3,3)*T]*noise;
xVehicleTrue(3) = AngleWrap(xVehicleTrue(3));

% This function simulates the odometry readings of the robot
function [uk, Qk]=get_odometry(StdOdometryNoise,xVehicleTrue, xVehicleTrueLast)
Qk = diag(StdOdometryNoise.^2);                     % odometry covariance matrix
DeltaX = xVehicleTrue - xVehicleTrueLast;
yaw=xVehicleTrueLast(3);
uk = [cos(yaw) sin(yaw) 0; -sin(yaw) cos(yaw)  0; 0 0 1] * DeltaX(1:3)+randn(3,1).*StdOdometryNoise/2;
uk(3) = AngleWrap(uk(3));

% This function simulates the range & bearing sensor
function [z, Rk, cFeature, mFeature] = get_measurements(Map, k, xVehicleTrue, SensorSettings)
global  FeatureVisibleStore
Rk = diag([SensorSettings.Rng_noise; deg2rad(SensorSettings.Brg_noise)]).^2;
done = 0;
Trys = 1;
z =[];
mFeature = -1;

while(~done && Trys <0.5*size(Map,2))

    %choose a random feature to see from True Map
    mFeature = ceil(size(Map,2)*rand(1));

    %this feature is observed from the vehicle as z=[range,angle]
    cFeature = Map(:,mFeature);
    z = DoObservationModel(xVehicleTrue,cFeature) + (sqrt(diag(Rk)).*randn);
    z(2) = AngleWrap(z(2));

    %look forward...and only up to Range and within the FieldOfView
    if k>SensorSettings.Faultingk0 && k<(SensorSettings.Faultingk0+SensorSettings.FaultGap)
        %forced failure in the sensor
        z =[];
        %FeatureVisibleStore(k)=1;
        done=1;
    else
        if(abs(pi/2-z(2))<SensorSettings.FieldOfView*pi/180 && z(1) < SensorSettings.Range)
            % if visible for the robot ...
            done =1 ;
            %FeatureVisibleStore(k)=0;
        else
            % try another one
            Trys =Trys+1;
            z =[];
            % FeatureVisibleStore(k)=1;
        end;
    end;
end;

% Given the robot and a feature position, both in the base frame, returns
% the range and bearing with which the robot can observe the feature
function [zOb] = DoObservationModel(x, Feature)
dis = Feature - x(1:2);
range = sqrt(dis(1)^2 + dis(2)^2);
bearing = atan2(dis(2), dis(1)) - x(3);
bearing = AngleWrap(bearing);
zOb = [range, bearing]';

% UTILITY FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ensures that any calculation with angles yields results between -pi and pi
function out = AngleWrap(in)
    out = in;
    if(in>pi)
        out = in - 2*pi;
    elseif(in<-pi)
        out = in + 2*pi;
    end;

% Compunding
function out = Compound(a, b)
    out = [cos(a(3)) * b(1)  - sin(a(3)) * b(2)  + a(1);
            sin(a(3)) * b(1)  + cos(a(3)) * b(2) + a(2);
            AngleWrap(a(3) + b(3))];

% GRAPHICS & PLOTTING %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare the graphics, plot the robot at the initial position with the
% initial uncertainty
function initial_graphics(Map,x,P,nSigmaRobot)
global hObsLine AxisSize handle;
close all;
figure(1);
hold on; grid off; axis equal;
plot(Map(1,:),Map(2,:),'g*');hold on;
set(gcf,'doublebuffer','on');
hObsLine = line([0,0],[0,0]);
set(hObsLine,'linestyle',':');
AxisSize = axis;
handle=plot_robot(x(1), x(2), x(3));
DoVehicleGraphics(x,P(1:2,1:2),nSigmaRobot);

% Draw the map with the robot and the currently observed feature
function draw_map(Map,x, P,nSigmaRobot, zk, mFeature, DrawEveryNFrames,k)
global hObsLine;

if(mod(k-2,DrawEveryNFrames)==0)
    DoVehicleGraphics(x,P(1:2,1:2),nSigmaRobot);
    if(~isempty(zk))
        set(hObsLine,'XData',[x(1),Map(1,mFeature)]);
        set(hObsLine,'YData',[x(2),Map(2,mFeature)]);
    end;
end
drawnow;

% Draw the robot and its uncertainty ellipse
function DoVehicleGraphics(x,P,nSigma)
h = PlotEllipse(x,P,nSigma);
if(~isempty(h))
    set(h,'color','b');
end;
plot_robot(x(1), x(2), x(3));

% Draw the ellipse corresponding to the covariance
function eH = PlotEllipse(x,P,nSigma)
eH = [];
P = P(1:2, 1:2);                    % only plot x-y part
x = x(1:2);
if(~any(diag(P) == 0))
    [V,D] = eig(P);
    y = nSigma * [cos(0:0.1:2*pi); sin(0:0.1:2*pi)];
    el = V * sqrtm(D) * y;
    el = [el el(:,1)] + repmat(x,1,size(el,2) + 1);
    eH = line(el(1,:),el(2,:));
end;

% Draw the robot
function [handle] = plot_robot(x, y, ang, handle)
%[handle] = plot_robot(x, y, ang)
%Creates a figure representing the ictineu auv and stores the coordinates
%in the global variable PLOT.ictineu_coord

%plot_robot(x, y, ang, handle)
%actualizes the ictineu auv plot with a new position x,y,ang

ang=ang+pi/2;
scaleFactor=3;

global PLOT

if nargin==3

    PLOT.ictineu_coord(1,:)=scaleFactor*[-0.3700    0.3700    0.3700   -0.3700   -0.3700    0.3700    0.3700   -0.3700    0.3579    0.3373    0.3052    0.2648    0.2200    0.1752    0.1348    0.1027...
        0.0821    0.0750    0.0821    0.1027    0.1348    0.1752    0.2200    0.2648    0.3052    0.3373    0.3579    0.3650   -0.3579   -0.3373   -0.3052   -0.2648...
        -0.2200   -0.1752   -0.1348   -0.1027   -0.0821   -0.0750   -0.0821   -0.1027   -0.1348   -0.1752   -0.2200   -0.2648   -0.3052   -0.3373   -0.3579   -0.3650...
        0.0713    0.0607    0.0441    0.0232         0   -0.0232   -0.0441   -0.0607   -0.0713   -0.0750   -0.0713   -0.0607   -0.0441   -0.0232         0    0.0232...
        0.0441    0.0607    0.0713    0.0750   -0.1500   -0.1732   -0.1941   -0.2107   -0.2213   -0.2250   -0.2213   -0.2107   -0.1941   -0.1732   -0.1500   -0.1000...
        -0.0951   -0.0809   -0.0588   -0.0309         0    0.0309    0.0588    0.0809    0.0951    0.1000    0.1500    0.1732    0.1941    0.2107    0.2213    0.2250...
        0.3700    0.3700   -0.3700   -0.3700    0.3700    0.3700    0.2250    0.2213    0.2107    0.1941    0.1732    0.1500    0.1000    0.0951    0.0809    0.0588...
        0.0309         0   -0.0309   -0.0588   -0.0809   -0.0951   -0.1000    0.1000    0.3000    0.3000    0.1000    0.1000    0.3000    0.3000    0.1000   -0.1000...
        -0.3000   -0.3000   -0.1000   -0.1000   -0.3000   -0.3000   -0.1000    0.2300    0.3300    0.3300    0.2300    0.3585    0.3543    0.3476    0.3393    0.3300...
        0.3207    0.3124    0.3057    0.3015    0.3000    0.3015    0.3057    0.3124    0.3207    0.3300    0.3393    0.3476    0.3543    0.3585    0.3600]*2;

    PLOT.ictineu_coord(2,:)=scaleFactor*[-0.2625   -0.2625   -0.2425   -0.2425    0.2625    0.2625    0.2425    0.2425    0.0448    0.0852    0.1173    0.1379    0.1450    0.1379    0.1173    0.0852...
        0.0448         0   -0.0448   -0.0852   -0.1173   -0.1379   -0.1450   -0.1379   -0.1173   -0.0852   -0.0448         0    0.0448    0.0852    0.1173    0.1379...
        0.1450    0.1379    0.1173    0.0852    0.0448         0   -0.0448   -0.0852   -0.1173   -0.1379   -0.1450   -0.1379   -0.1173   -0.0852   -0.0448    0.0000...
        0.0232    0.0441    0.0607    0.0713    0.0750    0.0713    0.0607    0.0441    0.0232         0   -0.0232   -0.0441   -0.0607   -0.0713   -0.0750   -0.0713...
        -0.0607   -0.0441   -0.0232         0    0.0750    0.0713    0.0607    0.0441    0.0232         0   -0.0232   -0.0441   -0.0607   -0.0713   -0.0750   -0.0750...
        -0.1059   -0.1338   -0.1559   -0.1701   -0.1750   -0.1701   -0.1559   -0.1338   -0.1059   -0.0750   -0.0750   -0.0713   -0.0607   -0.0441   -0.0232    0.0000...
        0   -0.2425   -0.2425    0.2425    0.2425         0         0    0.0232    0.0441    0.0607    0.0713    0.0750    0.0750    0.1059    0.1338    0.1559...
        0.1701    0.1750    0.1701    0.1559    0.1338    0.1059    0.0750    0.2425    0.2425    0.1700    0.1700   -0.2425   -0.2425   -0.1700   -0.1700    0.2425...
        0.2425    0.1700    0.1700   -0.2425   -0.2425   -0.1700   -0.1700   -0.0300   -0.0300    0.0300    0.0300    0.0093    0.0176    0.0243    0.0285    0.0300...
        0.0285    0.0243    0.0176    0.0093         0   -0.0093   -0.0176   -0.0243   -0.0285   -0.0300   -0.0285   -0.0243   -0.0176   -0.0093         0]*2;

    x=x+PLOT.ictineu_coord(1,:)*cos(ang)-PLOT.ictineu_coord(2,:)*sin(ang);
    y=y+PLOT.ictineu_coord(1,:)*sin(ang)+PLOT.ictineu_coord(2,:)*cos(ang);

    handle =  fill(x(1:4),y(1:4),[0.8 0.8 0.8],...
        x(5:8),y(5:8),[0.8 0.8 0.8],...
        x(9:28),y(9:28),'k',...
        x(29:48),y(29:48),'k',...
        x(49:68),y(49:68),[0.8 0.8 0.8],...
        x(69:119),y(69:119),'r',...
        x(120:123),y(120:123),[0.5 0 0],...
        x(124:127),y(124:127),[0.5 0 0],...
        x(128:131),y(128:131),[0.5 0 0],...
        x(132:135),y(132:135),[0.5 0 0],...
        x(136:139),y(136:139),[0.5 0.5 0.5],...
        x(140:159),y(140:159),'k');
    set(handle(3),'EdgeColor','none');
    set(handle(4),'EdgeColor','none');

elseif nargin==4

    x=x+PLOT.ictineu_coord(1,:)*cos(ang)-PLOT.ictineu_coord(2,:)*sin(ang);
    y=y+PLOT.ictineu_coord(1,:)*sin(ang)+PLOT.ictineu_coord(2,:)*cos(ang);

    set(handle(1),'XData',x(1:4),'YData',y(1:4));
    set(handle(2),'XData',x(5:8),'YData',y(5:8));
    set(handle(3),'XData',x(9:28),'YData',y(9:28));
    set(handle(4),'XData',x(29:48),'YData',y(29:48));
    set(handle(5),'XData',x(49:68),'YData',y(49:68));
    set(handle(6),'XData',x(69:119),'YData',y(69:119));
    set(handle(7),'XData',x(120:123),'YData',y(120:123));
    set(handle(8),'XData',x(124:127),'YData',y(124:127));
    set(handle(9),'XData',x(128:131),'YData',y(128:131));
    set(handle(10),'XData',x(132:135),'YData',y(132:135));
    set(handle(11),'XData',x(136:139),'YData',y(136:139));
    set(handle(12),'XData',x(140:159),'YData',y(140:159));

end

function DoGraphs(InnovStore,PStore,SStore,XErrStore,FeatureNotVisibleStore)

figure(2);
subplot(3,2,1);
plot(InnovStore(1,:));
hold on;
plot(SStore(1,:),'r');
plot(-SStore(1,:),'r');
title('Innovation');
ylabel('range (m)');

subplot(3,2,3);
plot(InnovStore(2,:)*180/pi);
hold on;
plot(SStore(2,:)*180/pi,'r');
plot(-SStore(2,:)*180/pi,'r')
ylabel('Bearing (deg)');

subplot(3,2,2);
plot(XErrStore(1,:));
hold on;
plot(3*PStore(1,:),'r');
plot(-3*PStore(1,:),'r');
title('Covariance and Error');
ylabel('x');

subplot(3,2,4);
plot(XErrStore(2,:));
hold on;
plot(3*PStore(2,:),'r');
plot(-3*PStore(2,:),'r')
ylabel('y');

subplot(3,2,6);
plot(XErrStore(3,:)*180/pi);
hold on;
plot(3*PStore(3,:)*180/pi,'r');
plot(-3*PStore(3,:)*180/pi,'r')
ylabel('Theta');
xlabel('time');

subplot(3,2,5);
plot(FeatureNotVisibleStore);
hold on;
title('Visible');
ylabel('no   -   yes');
xlabel('time');


% YOUR PROGRAMMING ASSIGMENT STARTS HERE %%%%%%%%%%%%%
% EKF equations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prediction of the robot movement.
function [xHat_B_kIk_1, PHat_B_kIk_1] = move_vehicle(x_B_kIk_1, P_B_kIk_1, uk, Qk)
    x_k_1 = x_B_kIk_1(1);
    y_k_1 = x_B_kIk_1(2);
    theta_k_1 = x_B_kIk_1(3);

    xR=uk(1);
    yR = uk(2);
    thetaR=uk(3);

    xHat_B_kIk_1= Compound(x_B_kIk_1, uk); 

    %Jacobians
    Ak = [  1 0 -sin(theta_k_1)*xR-cos(theta_k_1)*yR;...
            0 1 cos(theta_k_1)*xR-sin(theta_k_1)*yR;...
            0 0 1];
        
    Wk = [  cos(theta_k_1) -sin(theta_k_1) 0;...
            sin(theta_k_1) cos(theta_k_1) 0;...
            0 0 1];
        
    PHat_B_kIk_1 = Ak*P_B_kIk_1*Ak' + Wk*Qk*Wk';
    

% Update
function [x_B_k, P_B_k, Innovk, S] = update_position(xHat_B_kIk_1, PHat_B_kIk_1, zk, Rk, cFeature)

    xf = cFeature(1);
    yf = cFeature(2);
    xk = xHat_B_kIk_1(1);    
    yk = xHat_B_kIk_1(2);
    th_k = xHat_B_kIk_1(3);
    range = sqrt((xf-xk)^2+(yf-yk)^2);
    slope = (yf-yk)/(xf-xk);
    
    % evaluate the non-linear function at xHat, with vk=0
    h = [range;AngleWrap(atan2(yf-yk,xf-xk)-th_k)];
    
    % Jacobians
    Hk = [-(xf-xk)/range -(yf-yk)/range 0;
         1/(1+slope^2)*(yf-yk)/(xf-xk)^2, -1/(1+slope^2)/(xf-xk) -1];
    Vk = [1 0;0 1];
    
    % Innovation and its uncertainty
    Innovk=zk-h;
    Innovk(2) = AngleWrap(Innovk(2));

    S=Hk*PHat_B_kIk_1*Hk'+Rk;
    
    
    % Kalman gain
    Kk = PHat_B_kIk_1*Hk'*S^-1;
    x_B_k = xHat_B_kIk_1 + Kk*Innovk;
    P_B_k = (eye(3,3)-Kk*Hk)*PHat_B_kIk_1*(eye(3,3)-Kk*Hk)'+Kk*Rk*Kk';
       

% YOUR PROGRAMMING ASSIGMENT ENDS HERE %%%%%%%%%%%%%
