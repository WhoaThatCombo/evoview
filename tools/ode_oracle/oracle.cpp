// ODE oracle: runs a fixed scenario on the real ODE 0.15.2 (double
// precision) and prints every body's state each step as raw bits, so the
// Rust port (src/ode, test ode::oracle) can be compared bit for bit.
// Build: tools/ode_oracle/build.sh <ode-0.15.2 source dir>
// Keep this file and scenario() in src/ode/oracle.rs in step.
#include <ode/ode.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

static void hex(double v) {
    unsigned long long u;
    memcpy(&u, &v, 8);
    printf(" %016llx", u);
}
static void dump_body(dBodyID b) {
    const dReal *p = dBodyGetPosition(b), *q = dBodyGetQuaternion(b);
    const dReal *lv = dBodyGetLinearVel(b), *av = dBodyGetAngularVel(b);
    for (int i = 0; i < 3; i++) hex(p[i]);
    for (int i = 0; i < 4; i++) hex(q[i]);
    for (int i = 0; i < 3; i++) hex(lv[i]);
    for (int i = 0; i < 3; i++) hex(av[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 400;
    dInitODE2(0);
    dRandSetSeed(12345);
    dWorldID w = dWorldCreate();
    dWorldSetGravity(w, 0, -9.806, 0);
    dWorldSetERP(w, 0.25);
    dWorldSetCFM(w, 1e-7);
    dWorldSetQuickStepNumIterations(w, 24);
    dWorldSetContactMaxCorrectingVel(w, 5.0);
    dWorldSetContactSurfaceLayer(w, 0.001);
    dJointGroupID contacts = dJointGroupCreate(0);
    dBodyID B[12];
    dMass m;

    // 0: a tumbling box, asymmetric inertia (gyroscopic term), no joints
    B[0] = dBodyCreate(w);
    dMassSetBoxTotal(&m, 3.0, 0.4, 1.2, 0.2);
    dBodySetMass(B[0], &m);
    dBodySetPosition(B[0], 0, 5, 0);
    dBodySetAngularVel(B[0], 0.3, 7.0, 0.2);
    dBodySetLinearVel(B[0], 1, 0, 0);

    // 1-2: hinge to the world with limits + motor, ball joint below it
    B[1] = dBodyCreate(w);
    dMassSetCapsuleTotal(&m, 2.0, 3, 0.1, 0.8);
    dBodySetMass(B[1], &m);
    dBodySetPosition(B[1], 2, 3, 0.4);
    dMatrix3 R;
    dRFromAxisAndAngle(R, 0.2, 1, 0.3, 0.7);
    dBodySetRotation(B[1], R);
    dJointID h = dJointCreateHinge(w, 0);
    dJointAttach(h, B[1], 0);
    dJointSetHingeAnchor(h, 2, 3, 0);
    dJointSetHingeAxis(h, 1, 0, 0.2);
    dJointSetHingeParam(h, dParamLoStop, -0.6);
    dJointSetHingeParam(h, dParamHiStop, 0.9);
    dJointSetHingeParam(h, dParamVel, 1.5);
    dJointSetHingeParam(h, dParamFMax, 4.0);
    dJointSetHingeParam(h, dParamBounce, 0.3);
    dJointFeedback fb_h;
    dJointSetFeedback(h, &fb_h);
    B[2] = dBodyCreate(w);
    dMassSetSphereTotal(&m, 1.5, 0.15);
    dBodySetMass(B[2], &m);
    dBodySetPosition(B[2], 2.3, 2.2, 0.8);
    dJointID ball = dJointCreateBall(w, 0);
    dJointAttach(ball, B[1], B[2]);
    dJointSetBallAnchor(ball, 2.1, 2.6, 0.7);
    dJointSetBallParam(ball, dParamERP, 0.4);

    // 3-4: slider between two bodies with stops + motor, dball to the world
    B[3] = dBodyCreate(w);
    dMassSetBoxTotal(&m, 5.0, 0.5, 0.5, 0.5);
    dBodySetMass(B[3], &m);
    dBodySetPosition(B[3], -2, 4, 0);
    B[4] = dBodyCreate(w);
    dMassSetBoxTotal(&m, 1.0, 0.2, 0.2, 0.6);
    dBodySetMass(B[4], &m);
    dBodySetPosition(B[4], -2, 3.2, 0.3);
    dRFromEulerAngles(R, 0.1, -0.2, 0.3);
    dBodySetRotation(B[4], R);
    dJointID sl = dJointCreateSlider(w, 0);
    dJointAttach(sl, B[3], B[4]);
    dJointSetSliderAxis(sl, 0.1, 1, 0.05);
    dJointSetSliderParam(sl, dParamLoStop, -0.5);
    dJointSetSliderParam(sl, dParamHiStop, 0.1);
    dJointSetSliderParam(sl, dParamVel, -0.4);
    dJointSetSliderParam(sl, dParamFMax, 30.0);
    dJointFeedback fb_s;
    dJointSetFeedback(sl, &fb_s);
    dJointID db = dJointCreateDBall(w, 0);
    dJointAttach(db, B[3], 0);
    dJointSetDBallAnchor1(db, -2, 4.25, 0);
    dJointSetDBallAnchor2(db, -2.2, 5.5, 0.1);

    // 5-6: fixed joint pair, amotor (user, 3 axes) between them and an
    // Euler amotor on 6 to the world
    B[5] = dBodyCreate(w);
    dMassSetCylinderTotal(&m, 2.0, 2, 0.2, 0.5);
    dBodySetMass(B[5], &m);
    dBodySetPosition(B[5], 4, 6, -1);
    dBodySetAngularVel(B[5], 0, 2, 1);
    B[6] = dBodyCreate(w);
    dMassSetBoxTotal(&m, 1.0, 0.3, 0.3, 0.3);
    dMassTranslate(&m, 0.0, 0.0, 0.0);
    dBodySetMass(B[6], &m);
    dBodySetPosition(B[6], 4.2, 5.5, -1.1);
    dJointID fx = dJointCreateFixed(w, 0);
    dJointAttach(fx, B[5], B[6]);
    dJointSetFixed(fx);
    dJointSetFixedParam(fx, dParamCFM, 1e-5);
    dJointID am = dJointCreateAMotor(w, 0);
    dJointAttach(am, B[5], B[6]);
    dJointSetAMotorNumAxes(am, 3);
    dJointSetAMotorAxis(am, 0, 1, 1, 0, 0);
    dJointSetAMotorAxis(am, 1, 0, 0, 1, 0);
    dJointSetAMotorAxis(am, 2, 2, 0, 0, 1);
    dJointSetAMotorParam(am, dParamVel, 0.5);
    dJointSetAMotorParam(am, dParamFMax, 2.0);
    dJointSetAMotorParam(am, dParamVel3, -0.3);
    dJointSetAMotorParam(am, dParamFMax3, 1.0);
    dJointID ae = dJointCreateAMotor(w, 0);
    dJointAttach(ae, B[6], 0);
    dJointSetAMotorMode(ae, dAMotorEuler);
    dJointSetAMotorAxis(ae, 0, 1, 1, 0, 0);
    dJointSetAMotorAxis(ae, 2, 0, 0, 0, 1);
    dJointSetAMotorParam(ae, dParamLoStop, -0.3);
    dJointSetAMotorParam(ae, dParamHiStop, 0.3);
    dJointSetAMotorParam(ae, dParamLoStop2, -0.2);
    dJointSetAMotorParam(ae, dParamHiStop2, 0.2);
    dJointSetAMotorParam(ae, dParamFMax3, 0.5);

    // 7-8: boxes on a ground plane through contact joints (built each
    // step from the corners below y = 0): friction, approx1, bounce, soft
    B[7] = dBodyCreate(w);
    dMassSetBoxTotal(&m, 10.0, 1.0, 0.5, 1.5);
    dBodySetMass(B[7], &m);
    dBodySetPosition(B[7], 0, 0.6, 3);
    dRFromAxisAndAngle(R, 0.3, 0.1, 1, 0.4);
    dBodySetRotation(B[7], R);
    dBodySetLinearVel(B[7], 2, 0, -1);
    dBodySetAngularVel(B[7], 0, 1, 0);
    B[8] = dBodyCreate(w);
    dMassSetBoxTotal(&m, 4.0, 0.6, 0.6, 0.6);
    dBodySetMass(B[8], &m);
    dBodySetPosition(B[8], 3, 0.5, 3);
    dBodySetLinearVel(B[8], -3, 0, 0.5);
    dBodySetFiniteRotationMode(B[8], 1);
    dBodySetLinearDamping(B[8], 0.002);
    dBodySetAngularDamping(B[8], 0.01);

    // 9-10: two spheres meeting through a body-body contact
    B[9] = dBodyCreate(w);
    dMassSetSphereTotal(&m, 1.0, 0.3);
    dBodySetMass(B[9], &m);
    dBodySetPosition(B[9], -5, 0.3, -3);
    dBodySetLinearVel(B[9], 2, 0, 0.1);
    dBodySetGravityMode(B[9], 0);
    B[10] = dBodyCreate(w);
    dMassSetSphereTotal(&m, 2.0, 0.4);
    dBodySetMass(B[10], &m);
    dBodySetPosition(B[10], -3, 0.35, -3);
    dBodySetGravityMode(B[10], 0);
    dBodySetAngularVel(B[10], 0, 0, 3);

    const double half[2][3] = {{0.5, 0.25, 0.75}, {0.3, 0.3, 0.3}};
    const double dt = 0.003;
    for (int s = 0; s < steps; s++) {
        // forces from the caller, as a game applies them
        dBodyAddForce(B[3], 0, 0, 5.0 * sin(s * 0.01));
        dBodyAddRelTorque(B[5], 0.1, 0, 0);
        dBodyAddForceAtRelPos(B[7], 0, 0, 3.0, 0.3, 0.2, 0.0);
        dJointAddHingeTorque(h, 0.5);
        for (int k = 0; k < 2; k++) {
            dBodyID b = B[7 + k];
            for (int c = 0; c < 8; c++) {
                dVector3 p;
                dBodyGetRelPointPos(b, (c & 1 ? 1 : -1) * half[k][0], (c & 2 ? 1 : -1) * half[k][1], (c & 4 ? 1 : -1) * half[k][2], p);
                if (p[1] < 0) {
                    dContact ct;
                    memset(&ct, 0, sizeof(ct));
                    ct.surface.mode = dContactApprox1 | dContactSoftERP | dContactSoftCFM | dContactBounce;
                    ct.surface.mu = k == 0 ? 0.8 : 0.3;
                    ct.surface.bounce = 0.2;
                    ct.surface.bounce_vel = 0.1;
                    ct.surface.soft_erp = 0.3;
                    ct.surface.soft_cfm = 1e-5;
                    ct.geom.pos[0] = p[0];
                    ct.geom.pos[1] = p[1];
                    ct.geom.pos[2] = p[2];
                    ct.geom.normal[1] = 1;
                    ct.geom.depth = -p[1];
                    dJointID cj = dJointCreateContact(w, contacts, &ct);
                    dJointAttach(cj, b, 0);
                }
            }
        }
        {
            const dReal *p1 = dBodyGetPosition(B[9]), *p2 = dBodyGetPosition(B[10]);
            dVector3 d = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
            dReal len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            if (len < 0.7) {
                dContact ct;
                memset(&ct, 0, sizeof(ct));
                ct.surface.mode = dContactMu2 | dContactFDir1 | dContactSlip1;
                ct.surface.mu = 0.5;
                ct.surface.mu2 = 0.2;
                ct.surface.slip1 = 0.01;
                ct.geom.normal[0] = -d[0] / len;
                ct.geom.normal[1] = -d[1] / len;
                ct.geom.normal[2] = -d[2] / len;
                ct.geom.pos[0] = p1[0] + d[0] / len * 0.3;
                ct.geom.pos[1] = p1[1] + d[1] / len * 0.3;
                ct.geom.pos[2] = p1[2] + d[2] / len * 0.3;
                ct.geom.depth = 0.7 - len;
                ct.fdir1[0] = 0;
                ct.fdir1[1] = 0;
                ct.fdir1[2] = 1;
                dJointID cj = dJointCreateContact(w, contacts, &ct);
                dJointAttach(cj, B[9], B[10]);
            }
        }
        dWorldQuickStep(w, dt);
        dJointGroupEmpty(contacts);
        printf("S%d", s);
        hex(dJointGetHingeAngle(h));
        hex(dJointGetSliderPosition(sl));
        for (int i = 0; i < 3; i++) hex(fb_h.f1[i]);
        for (int i = 0; i < 3; i++) hex(fb_s.t2[i]);
        hex(dJointGetAMotorAngle(ae, 1));
        printf("\n");
        for (int i = 0; i < 11; i++) dump_body(B[i]);
    }
    dWorldDestroy(w);
    dCloseODE();
    return 0;
}
