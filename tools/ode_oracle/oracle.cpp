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

static int collide_scenario(int steps);
// the collision scenario's top space: 0 hash, 1 quadtree, 2 sweep and prune
static int space_kind;
static int trimesh_scenario(int steps);
static int shapes_scenario(int steps);
static int ttl_scenario(int steps);
static int joints_scenario(int steps);

int main(int argc, char **argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 400;
    if (argc > 2 && !strcmp(argv[2], "collide")) return collide_scenario(steps);
    if (argc > 2 && !strcmp(argv[2], "trimesh")) return trimesh_scenario(steps);
    if (argc > 2 && !strcmp(argv[2], "shapes")) return shapes_scenario(steps);
    if (argc > 2 && !strcmp(argv[2], "ttl")) return ttl_scenario(steps);
    if (argc > 2 && !strcmp(argv[2], "joints")) return joints_scenario(steps);
    if (argc > 2 && !strcmp(argv[2], "quadtree")) { space_kind = 1; return collide_scenario(steps); }
    if (argc > 2 && !strcmp(argv[2], "sap")) { space_kind = 2; return collide_scenario(steps); }
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

// ---- scenario 2: collision. A pile of boxes, spheres and capsules on a
// plane in a hash space, the spheres in a nested simple space, an offset
// capsule, and a ray on a body; contacts from dCollide each step.
static dWorldID cw;
static dJointGroupID cgroup;
static int ray_hits;
static int maxc = 8;
static void dump_contacts(dContactGeom *c, int n) {
    for (int i = 0; i < n; i++) {
        printf("C");
        for (int k = 0; k < 3; k++) hex(c[i].pos[k]);
        for (int k = 0; k < 3; k++) hex(c[i].normal[k]);
        hex(c[i].depth);
        printf("\n");
    }
}
static void near_cb(void *, dGeomID o1, dGeomID o2) {
    if (dGeomIsSpace(o1) || dGeomIsSpace(o2)) {
        dSpaceCollide2(o1, o2, 0, &near_cb);
        return;
    }
    dContactGeom cg[8];
    int n = dCollide(o1, o2, maxc, cg, sizeof(dContactGeom));
    dump_contacts(cg, n);
    if (dGeomGetClass(o1) == dRayClass || dGeomGetClass(o2) == dRayClass) {
        ray_hits += n;
        return;
    }
    for (int i = 0; i < n; i++) {
        dContact ct;
        memset(&ct, 0, sizeof(ct));
        ct.surface.mode = dContactApprox1 | dContactBounce | dContactSoftCFM;
        ct.surface.mu = 0.6;
        ct.surface.bounce = 0.1;
        ct.surface.bounce_vel = 0.2;
        ct.surface.soft_cfm = 1e-6;
        ct.geom = cg[i];
        dJointID j = dJointCreateContact(cw, cgroup, &ct);
        dJointAttach(j, dGeomGetBody(o1), dGeomGetBody(o2));
    }
}
static int collide_scenario(int steps) {
    dInitODE2(0);
    dRandSetSeed(777);
    cw = dWorldCreate();
    dWorldSetGravity(cw, 0, -9.806, 0);
    dWorldSetQuickStepNumIterations(cw, 20);
    dWorldSetContactSurfaceLayer(cw, 0.002);
    cgroup = dJointGroupCreate(0);
    dSpaceID space;
    if (space_kind == 1) {
        dVector3 c = {0, 2, 0}, e = {4, 4, 4};
        space = dQuadTreeSpaceCreate(0, c, e, 4);
    } else if (space_kind == 2) {
        space = dSweepAndPruneSpaceCreate(0, dSAP_AXES_XZY);
    } else {
        space = dHashSpaceCreate(0);
        dHashSpaceSetLevels(space, -4, 6);
    }
    dSpaceID inner = dSimpleSpaceCreate(space);
    dCreatePlane(space, 0, 1, 0, 0);
    dBodyID B[16];
    int nb = 0;
    dMass m;
    dMatrix3 R;
    for (int i = 0; i < 6; i++) {
        dBodyID b = dBodyCreate(cw);
        dMassSetBoxTotal(&m, 2.0 + i, 0.4 + 0.05 * i, 0.3, 0.5);
        dBodySetMass(b, &m);
        dBodySetPosition(b, 0.3 * i - 0.8, 0.5 + 0.45 * i, 0.1 * (i % 3));
        dRFromAxisAndAngle(R, 0.3 * i, 1, 0.2, 0.25 * i);
        dBodySetRotation(b, R);
        dBodySetAngularVel(b, 0.2 * i, 0, -0.1 * i);
        dGeomID g = dCreateBox(space, 0.4 + 0.05 * i, 0.3, 0.5);
        dGeomSetBody(g, b);
        B[nb++] = b;
    }
    for (int i = 0; i < 4; i++) {
        dBodyID b = dBodyCreate(cw);
        dMassSetSphereTotal(&m, 1.0, 0.15 + 0.03 * i);
        dBodySetMass(b, &m);
        dBodySetPosition(b, -0.5 + 0.35 * i, 3.2 + 0.3 * i, -0.2 + 0.15 * i);
        dBodySetLinearVel(b, 0.3, -1, 0.1 * i);
        dGeomID g = dCreateSphere(inner, 0.15 + 0.03 * i);
        dGeomSetBody(g, b);
        B[nb++] = b;
    }
    for (int i = 0; i < 3; i++) {
        dBodyID b = dBodyCreate(cw);
        dMassSetCapsuleTotal(&m, 1.5, 3, 0.1, 0.5);
        dBodySetMass(b, &m);
        dBodySetPosition(b, 0.6 - 0.4 * i, 4.5 + 0.4 * i, 0.3);
        dRFromAxisAndAngle(R, 1, 0.1 * i, 0, 1.2 + 0.3 * i);
        dBodySetRotation(b, R);
        dGeomID g = dCreateCapsule(space, 0.1, 0.5);
        dGeomSetBody(g, b);
        if (i == 1) {
            dGeomSetOffsetPosition(g, 0.05, 0, 0.02);
            dRFromAxisAndAngle(R, 0, 1, 0, 0.3);
            dGeomSetOffsetRotation(g, R);
        }
        B[nb++] = b;
    }
    dGeomID ray = dCreateRay(space, 2.0);
    dGeomSetBody(ray, B[0]);
    dGeomSetOffsetRotation(ray, R);
    for (int s = 0; s < steps; s++) {
        ray_hits = 0;
        maxc = s % 3 + 2;
        printf("S%d\n", s);
        dSpaceCollide(space, 0, &near_cb);
        dSpaceCollide(inner, 0, &near_cb);
        dWorldQuickStep(cw, 0.003);
        dJointGroupEmpty(cgroup);
        printf("R%d\n", ray_hits);
        for (int i = 0; i < nb; i++) dump_body(B[i]);
    }
    return 0;
}

// ---- scenario 3: trimesh queries. A bumpy grid mesh (double data) that
// turns a little each step, probed by a fan of rays with every flag mix.
static double tm_verts[13 * 13 * 3];
static dTriIndex tm_idx[12 * 12 * 6];
static void build_grid() {
    int k = 0;
    for (int j = 0; j < 13; j++)
        for (int i = 0; i < 13; i++) {
            double x = (i - 6) * 0.5, z = (j - 6) * 0.5;
            tm_verts[k++] = x;
            tm_verts[k++] = 0.3 * sin(x * 1.3) * cos(z * 0.7) + 0.1 * x;
            tm_verts[k++] = z;
        }
    k = 0;
    for (int j = 0; j < 12; j++)
        for (int i = 0; i < 12; i++) {
            int a = j * 13 + i, b = a + 1, c = a + 13, d = c + 1;
            if ((i + j) & 1) {
                tm_idx[k++] = a; tm_idx[k++] = c; tm_idx[k++] = b;
                tm_idx[k++] = b; tm_idx[k++] = c; tm_idx[k++] = d;
            } else {
                tm_idx[k++] = a; tm_idx[k++] = c; tm_idx[k++] = d;
                tm_idx[k++] = a; tm_idx[k++] = d; tm_idx[k++] = b;
            }
        }
}
static int trimesh_scenario(int steps) {
    dInitODE2(0);
    build_grid();
    dTriMeshDataID data = dGeomTriMeshDataCreate();
    dGeomTriMeshDataBuildDouble(data, tm_verts, 3 * sizeof(double), 13 * 13, tm_idx, 12 * 12 * 6, 3 * sizeof(dTriIndex));
    dSpaceID space = dSimpleSpaceCreate(0);
    dGeomID mesh = dCreateTriMesh(space, data, 0, 0, 0);
    dGeomID ray = dCreateRay(space, 1.0);
    dGeomID sph = dCreateSphere(space, 0.3);
    dGeomID bx = dCreateBox(space, 0.6, 0.3, 0.9);
    dGeomID cap = dCreateCapsule(space, 0.2, 0.6);
    dGeomID pln = dCreatePlane(0, 0, 1, 0, 0);
    dMatrix3 R;
    for (int s = 0; s < steps; s++) {
        printf("S%d\n", s);
        dRFromAxisAndAngle(R, 0.1, 1, 0.2, 0.05 * s);
        dGeomSetRotation(mesh, R);
        dGeomSetPosition(mesh, 0.01 * s, -0.2 + 0.003 * s, 0.02 * s);
        dReal aabb[6];
        dGeomGetAABB(mesh, aabb);
        for (int k = 0; k < 6; k++) hex(aabb[k]);
        printf("\n");
        for (int r = 0; r < 24; r++) {
            double ox = -2.7 + 0.23 * r + 0.01 * s, oz = 2.5 - 0.21 * r;
            double dx = 0.2 * sin(r * 0.7), dz = 0.15 * cos(r * 1.1);
            double oy = (r % 5 == 4) ? -1.5 : 1.5;
            double dy = (r % 5 == 4) ? 1.0 : -1.0;
            if (r % 7 == 3) { dx = 1.0; dy = -0.05; dz = 0.3; oy = 0.1; ox = -3.5; }
            dGeomRaySetLength(ray, (r % 3 == 0) ? 1e30 : 1.6 + 0.4 * (r % 4));
            dGeomRaySetParams(ray, (r >> 1) & 1, (r >> 2) & 1);
            dGeomRaySetClosestHit(ray, (r % 6 == 5) && !((r >> 1) & 1));
            if (r % 7 == 3) {
                dGeomRaySetLength(ray, r == 17 ? 4.5 : 1e30);
                dGeomRaySetParams(ray, 0, 0);
                dGeomRaySetClosestHit(ray, 0);
            }
            dGeomRaySet(ray, ox, oy, oz, dx, dy, dz);
            int maxc = 1 + (r % 4) * 3;
            dContactGeom cg[16];
            int n = dCollide(ray, mesh, maxc, cg, sizeof(dContactGeom));
            printf("A%d %d", r, n);
            for (int i = 0; i < n; i++) printf(" %d/%d", cg[i].side1, cg[i].side2);
            printf("\n");
            dump_contacts(cg, n);
            n = dCollide(mesh, ray, maxc, cg, sizeof(dContactGeom));
            printf("B%d %d", r, n);
            for (int i = 0; i < n; i++) printf(" %d/%d", cg[i].side1, cg[i].side2);
            printf("\n");
            dump_contacts(cg, n);
        }
        for (int q = 0; q < 20; q++) {
            double x = -2.6 + 0.27 * q, z = 2.2 - 0.23 * q + 0.005 * s;
            double y = 0.25 * sin(q * 0.9 + s * 0.05) + 0.05 * x;
            dGeomSphereSetRadius(sph, 0.15 + 0.05 * (q % 6));
            dGeomSetPosition(sph, x, y, z);
            int maxc = 1 + (q % 5) * 3;
            dContactGeom cg[16];
            int n = dCollide(sph, mesh, maxc, cg, sizeof(dContactGeom));
            printf("P%d %d", q, n);
            for (int i = 0; i < n; i++) printf(" %d/%d", cg[i].side1, cg[i].side2);
            printf("\n");
            dump_contacts(cg, n);
            n = dCollide(mesh, sph, maxc, cg, sizeof(dContactGeom));
            printf("Q%d %d\n", q, n);
            dump_contacts(cg, n);
        }
        for (int q = 0; q < 16; q++) {
            double x = -2.4 + 0.31 * q, z = 2.0 - 0.27 * q + 0.004 * s;
            double y = 0.2 * sin(q * 1.3 + s * 0.04) + 0.05 * x;
            dGeomBoxSetLengths(bx, 0.3 + 0.15 * (q % 4), 0.2 + 0.1 * (q % 3), 0.4 + 0.2 * (q % 5));
            dMatrix3 RB;
            dRFromAxisAndAngle(RB, sin(q * 0.5), 1.0, cos(q * 0.3), 0.4 * q + 0.03 * s);
            dGeomSetRotation(bx, RB);
            dGeomSetPosition(bx, x, y, z);
            int maxc = 1 + (q % 5) * 3;
            dContactGeom cg[16];
            int n = dCollide(bx, mesh, maxc | ((q % 7 == 6) ? CONTACTS_UNIMPORTANT : 0), cg, sizeof(dContactGeom));
            printf("X%d %d", q, n);
            for (int i = 0; i < n; i++) printf(" %d/%d", cg[i].side1, cg[i].side2);
            printf("\n");
            dump_contacts(cg, n);
            n = dCollide(mesh, bx, maxc, cg, sizeof(dContactGeom));
            printf("Y%d %d\n", q, n);
            dump_contacts(cg, n);
        }
        for (int q = 0; q < 16; q++) {
            double x = -2.5 + 0.33 * q, z = -2.1 + 0.26 * q - 0.004 * s;
            double y = 0.22 * cos(q * 1.1 + s * 0.05) + 0.05 * x;
            dGeomCapsuleSetParams(cap, 0.1 + 0.05 * (q % 4), 0.3 + 0.25 * (q % 3));
            dMatrix3 RC;
            dRFromAxisAndAngle(RC, cos(q * 0.7), 0.4, sin(q * 0.2), 0.5 * q + 0.02 * s);
            dGeomSetRotation(cap, RC);
            dGeomSetPosition(cap, x, y, z);
            int maxc = 1 + (q % 5) * 3;
            dContactGeom cg[16];
            int n = dCollide(cap, mesh, maxc | ((q % 6 == 5) ? CONTACTS_UNIMPORTANT : 0), cg, sizeof(dContactGeom));
            printf("K%d %d", q, n);
            for (int i = 0; i < n; i++) printf(" %d/%d", cg[i].side1, cg[i].side2);
            printf("\n");
            dump_contacts(cg, n);
            n = dCollide(mesh, cap, maxc, cg, sizeof(dContactGeom));
            printf("L%d %d\n", q, n);
            dump_contacts(cg, n);
        }
        for (int q = 0; q < 4; q++) {
            dGeomPlaneSetParams(pln, 0.1 * sin(q + s * 0.1), 1.0, 0.2 * cos(q * 2.0), -0.15 + 0.1 * q);
            int maxc = 2 + q * 3;
            dContactGeom cg[16];
            int n = dCollide(mesh, pln, maxc, cg, sizeof(dContactGeom));
            printf("M%d %d", q, n);
            for (int i = 0; i < n; i++) printf(" %d/%d", cg[i].side1, cg[i].side2);
            printf("\n");
            dump_contacts(cg, n);
            n = dCollide(pln, mesh, maxc, cg, sizeof(dContactGeom));
            printf("N%d %d\n", q, n);
            dump_contacts(cg, n);
        }
    }
    return 0;
}

// ---- scenario 4: the other shapes. Heightfields (placeable and not),
// cylinders, convexes and a geom transform, queried with dCollide both
// ways against every partner they have a collider for.
static dGeomID SG[16];
static int nsg;
static int gi(dGeomID g) {
    for (int i = 0; i < nsg; i++)
        if (SG[i] == g) return i;
    return -1;
}
static void query(char tag, int q, dGeomID a, dGeomID b, int maxc) {
    dContactGeom cg[16];
    // convex-convex leaves side1/side2 as it found them: start from -1
    for (int i = 0; i < 16; i++) cg[i].side1 = cg[i].side2 = -1;
    int n = dCollide(a, b, maxc, cg, sizeof(dContactGeom));
    printf("%c%d %d", tag, q, n);
    for (int i = 0; i < n; i++) printf(" %d/%d/%d/%d", cg[i].side1, cg[i].side2, gi(cg[i].g1), gi(cg[i].g2));
    printf("\n");
    dump_contacts(cg, n);
}
static void both(char tag, int q, dGeomID a, dGeomID b, int maxc) {
    query(tag, q, a, b, maxc);
    query(tag + 1, q, b, a, maxc);
}
// an octahedron with half-axes a, b, c
static void octa(double a, double b, double c, double *planes, double *points, unsigned *polys, dTriIndex *tris) {
    double p[6][3] = {{a, 0, 0}, {-a, 0, 0}, {0, b, 0}, {0, -b, 0}, {0, 0, c}, {0, 0, -c}};
    for (int i = 0; i < 6; i++)
        for (int k = 0; k < 3; k++) points[i * 3 + k] = p[i][k];
    double l = sqrt(1 / (a * a) + 1 / (b * b) + 1 / (c * c));
    int f = 0;
    for (int sx = 1; sx >= -1; sx -= 2)
        for (int sy = 1; sy >= -1; sy -= 2)
            for (int sz = 1; sz >= -1; sz -= 2) {
                planes[f * 4 + 0] = sx / a / l;
                planes[f * 4 + 1] = sy / b / l;
                planes[f * 4 + 2] = sz / c / l;
                planes[f * 4 + 3] = 1 / l;
                int X = sx > 0 ? 0 : 1, Y = sy > 0 ? 2 : 3, Z = sz > 0 ? 4 : 5;
                int v1 = Y, v2 = Z;
                if (sx * sy * sz < 0) { v1 = Z; v2 = Y; }
                polys[f * 4] = 3;
                polys[f * 4 + 1] = X;
                polys[f * 4 + 2] = v1;
                polys[f * 4 + 3] = v2;
                tris[f * 3] = X;
                tris[f * 3 + 1] = v1;
                tris[f * 3 + 2] = v2;
                f++;
            }
}
static double hf_h[9 * 11];
static double cv_planes[32], cv_points[18], om_verts[18];
static unsigned cv_polys[32];
static dTriIndex om_tris[24];
static int shapes_scenario(int steps) {
    dInitODE2(0);
    build_grid();
    dTriMeshDataID data = dGeomTriMeshDataCreate();
    dGeomTriMeshDataBuildDouble(data, tm_verts, 3 * sizeof(double), 13 * 13, tm_idx, 12 * 12 * 6, 3 * sizeof(dTriIndex));
    octa(0.5, 0.35, 0.4, cv_planes, cv_points, cv_polys, om_tris);
    for (int i = 0; i < 18; i++) om_verts[i] = cv_points[i] * 0.8;
    dTriMeshDataID odata = dGeomTriMeshDataCreate();
    dGeomTriMeshDataBuildDouble(odata, om_verts, 3 * sizeof(double), 6, om_tris, 24, 3 * sizeof(dTriIndex));
    for (int j = 0; j < 11; j++)
        for (int i = 0; i < 9; i++) hf_h[i + j * 9] = 0.25 * sin(0.9 * i) * cos(0.6 * j) + 0.05 * j;
    dHeightfieldDataID hd = dGeomHeightfieldDataCreate();
    dGeomHeightfieldDataBuildDouble(hd, hf_h, 1, 4.0, 5.0, 9, 11, 1.0, 0.0, 0.5, 0);
    dGeomID sph = SG[nsg++] = dCreateSphere(0, 0.3);
    dGeomID bx = SG[nsg++] = dCreateBox(0, 0.6, 0.3, 0.9);
    dGeomID cap = SG[nsg++] = dCreateCapsule(0, 0.2, 0.6);
    dGeomID cyl = SG[nsg++] = dCreateCylinder(0, 0.25, 0.7);
    dGeomID cvx = SG[nsg++] = dCreateConvex(0, cv_planes, 8, cv_points, 6, cv_polys);
    dGeomID cvx2 = SG[nsg++] = dCreateConvex(0, cv_planes, 8, cv_points, 6, cv_polys);
    dGeomID ray = SG[nsg++] = dCreateRay(0, 2.5);
    dGeomID pln = SG[nsg++] = dCreatePlane(0, 0, 1, 0, 0);
    dGeomID mesh = SG[nsg++] = dCreateTriMesh(0, data, 0, 0, 0);
    dGeomID om = SG[nsg++] = dCreateTriMesh(0, odata, 0, 0, 0);
    dGeomID hfn = SG[nsg++] = dCreateHeightfield(0, hd, 0);
    dGeomID hfp = SG[nsg++] = dCreateHeightfield(0, hd, 1);
    dGeomID tr = SG[nsg++] = dCreateGeomTransform(0);
    dGeomID tro = SG[nsg++] = dCreateBox(0, 0.5, 0.2, 0.4);
    dMatrix3 R;
    dGeomSetPosition(tro, 0.1, 0.05, -0.1);
    dRFromAxisAndAngle(R, 1, 0, 0, 0.3);
    dGeomSetRotation(tro, R);
    dGeomTransformSetGeom(tr, tro);
    dGeomSetPosition(mesh, 0, -0.2, 0);
    dGeomID objs[7] = {sph, bx, cap, cyl, cvx, ray, om};
    for (int s = 0; s < steps; s++) {
        printf("S%d\n", s);
        dRFromAxisAndAngle(R, 0.2, 1, 0.1, 0.02 * s);
        dGeomSetRotation(hfp, R);
        dGeomSetPosition(hfp, 0.3, -0.1 + 0.002 * s, 0.2);
        for (int q = 0; q < 14; q++) {
            dGeomID o = objs[q % 7];
            double x = -1.8 + 0.27 * q + 0.01 * s, z = -2.2 + 0.33 * q;
            double y = 0.3 * sin(q + 0.05 * s) + 0.1 + 0.05 * z;
            dRFromAxisAndAngle(R, sin(q), 1, cos(q), 0.3 * q + 0.02 * s);
            if (o == ray) dGeomRaySet(ray, x, y + 1, z, 0.1 * sin(q), -1, 0.05);
            else {
                dGeomSetRotation(o, R);
                dGeomSetPosition(o, x, y, z);
            }
            int maxc = 1 + (q % 5) * 3;
            both('H', q, hfn, o, maxc);
            both('J', q, hfp, o, maxc);
        }
        for (int q = 0; q < 10; q++) {
            dRFromAxisAndAngle(R, cos(0.7 * q), 0.5, sin(0.3 * q), 0.4 * q + 0.03 * s);
            dGeomSetRotation(cyl, R);
            double x = -1 + 0.2 * q, z = 1 - 0.15 * q;
            double y = 0.15 * sin(q * 0.8 + s * 0.04);
            dGeomSetPosition(cyl, x, y, z);
            dGeomID p = 0;
            switch (q % 5) {
            case 0: p = sph; dGeomSetPosition(sph, x + 0.25 * sin(q + 0.1 * s), y + 0.3, z + 0.1); break;
            case 1:
                p = bx;
                dRFromAxisAndAngle(R, 1, sin(q), 0.2, 0.5 * q - 0.02 * s);
                dGeomSetRotation(bx, R);
                dGeomSetPosition(bx, x + 0.2, y + 0.35 + 0.1 * cos(0.05 * s), z - 0.1);
                break;
            case 2: p = pln; dGeomPlaneSetParams(pln, 0.1 * sin(q + 0.1 * s), 1, 0.1 * cos(q), y - 0.2); break;
            case 3:
                p = ray;
                dGeomRaySet(ray, x + 0.1 * sin(0.3 * s), y + 1.2, z - 0.05, 0.05 * q - 0.2, -1, 0.1);
                break;
            case 4: p = mesh; dGeomSetPosition(cyl, x, 0.3 * sin(x * 1.3) * cos(z * 0.7) + 0.1 * x - 0.1, z); break;
            }
            int maxc = 1 + (q % 4) * 3;
            both('Y', q, cyl, p, maxc);
        }
        for (int q = 0; q < 12; q++) {
            dRFromAxisAndAngle(R, sin(0.9 * q), 1, 0.3, 0.35 * q + 0.025 * s);
            dGeomSetRotation(cvx, R);
            double x = 0.5 - 0.1 * q, z = -0.5 + 0.12 * q;
            double y = 0.2 * cos(q * 0.6 + s * 0.05);
            dGeomSetPosition(cvx, x, y, z);
            dGeomID p = 0;
            switch (q % 6) {
            case 0: p = pln; dGeomPlaneSetParams(pln, 0.15 * cos(q + 0.07 * s), 1, 0.1 * sin(q), y - 0.25); break;
            case 1: p = sph; dGeomSetPosition(sph, x + 0.3 * sin(0.1 * s + q), y + 0.4, z + 0.2 * cos(0.13 * s)); break;
            case 2:
                p = cvx2;
                dRFromAxisAndAngle(R, 0.3, cos(q), 1, -0.2 * q + 0.04 * s);
                dGeomSetRotation(cvx2, R);
                dGeomSetPosition(cvx2, x + 0.3 + 0.2 * sin(0.06 * s), y + 0.35, z - 0.15);
                break;
            case 3: p = ray; dGeomRaySet(ray, x - 1.2, y + 0.05 * sin(0.2 * s), z + 0.1, 1, -0.02 * q, 0.1); break;
            case 4: p = bx; dGeomSetPosition(bx, x + 0.2, y + 0.3, z); break;
            case 5: p = mesh; break;
            }
            int maxc = 1 + (q % 4) * 3;
            both('V', q, cvx, p, maxc);
        }
        for (int q = 0; q < 8; q++) {
            dRFromAxisAndAngle(R, 0.5, 1, sin(q), 0.3 * q + 0.03 * s);
            dGeomSetRotation(tr, R);
            double x = -0.3 + 0.1 * q, z = 0.2 * sin(q), y = 0.1 * cos(q + 0.05 * s);
            dGeomSetPosition(tr, x, y, z);
            dGeomTransformSetInfo(tr, q & 1);
            dGeomID p = 0;
            switch (q % 4) {
            case 0: p = sph; dGeomSetPosition(sph, x + 0.2, y + 0.3, z + 0.1 * sin(0.1 * s)); break;
            case 1: p = pln; dGeomPlaneSetParams(pln, 0, 1, 0.1 * sin(0.05 * s), y - 0.1); break;
            case 2: p = cvx; dGeomSetPosition(cvx, x, y + 0.45, z + 0.1); break;
            case 3: p = cyl; dGeomSetPosition(cyl, x - 0.1, y + 0.3, z); break;
            }
            int maxc = 1 + (q % 4) * 3;
            both('T', q, tr, p, maxc);
        }
    }
    return 0;
}

// ---- scenario 5: trimesh against trimesh. The bumpy grid against a
// smaller bumpy grid and an octahedron, posed across its surface.
static double g2_verts[7 * 7 * 3];
static dTriIndex g2_idx[6 * 6 * 6];
static int ttl_scenario(int steps) {
    dInitODE2(0);
    build_grid();
    int k = 0;
    for (int j = 0; j < 7; j++)
        for (int i = 0; i < 7; i++) {
            double x = (i - 3) * 0.3, z = (j - 3) * 0.3;
            g2_verts[k++] = x;
            g2_verts[k++] = 0.15 * cos(x * 2.1) * sin(z * 1.7);
            g2_verts[k++] = z;
        }
    k = 0;
    for (int j = 0; j < 6; j++)
        for (int i = 0; i < 6; i++) {
            int a = j * 7 + i, b = a + 1, c = a + 7, d = c + 1;
            g2_idx[k++] = a; g2_idx[k++] = c; g2_idx[k++] = d;
            g2_idx[k++] = a; g2_idx[k++] = d; g2_idx[k++] = b;
        }
    octa(0.5, 0.35, 0.4, cv_planes, cv_points, cv_polys, om_tris);
    for (int i = 0; i < 18; i++) om_verts[i] = cv_points[i] * 0.8;
    dTriMeshDataID da = dGeomTriMeshDataCreate();
    dGeomTriMeshDataBuildDouble(da, tm_verts, 3 * sizeof(double), 13 * 13, tm_idx, 12 * 12 * 6, 3 * sizeof(dTriIndex));
    dTriMeshDataID db = dGeomTriMeshDataCreate();
    dGeomTriMeshDataBuildDouble(db, g2_verts, 3 * sizeof(double), 7 * 7, g2_idx, 6 * 6 * 6, 3 * sizeof(dTriIndex));
    dTriMeshDataID dc = dGeomTriMeshDataCreate();
    dGeomTriMeshDataBuildDouble(dc, om_verts, 3 * sizeof(double), 6, om_tris, 24, 3 * sizeof(dTriIndex));
    nsg = 0;
    dGeomID A = SG[nsg++] = dCreateTriMesh(0, da, 0, 0, 0);
    dGeomID B = SG[nsg++] = dCreateTriMesh(0, db, 0, 0, 0);
    dGeomID O = SG[nsg++] = dCreateTriMesh(0, dc, 0, 0, 0);
    dMatrix3 R;
    for (int s = 0; s < steps; s++) {
        printf("S%d\n", s);
        dRFromAxisAndAngle(R, 0.1, 1, 0.2, 0.03 * s);
        dGeomSetRotation(A, R);
        dGeomSetPosition(A, 0.005 * s, -0.1, 0.01 * s);
        for (int q = 0; q < 10; q++) {
            double x = -2.2 + 0.45 * q + 0.01 * s, z = 1.9 - 0.4 * q;
            double y = 0.3 * sin(x * 1.3) * cos(z * 0.7) + 0.1 * x - 0.1 + 0.08 * sin(q + 0.1 * s);
            int maxc = (1 + (q % 5) * 3) | ((q % 4 == 3) ? CONTACTS_UNIMPORTANT : 0);
            dRFromAxisAndAngle(R, sin(q), 1, 0.4 * cos(q), 0.2 * q + 0.04 * s);
            dGeomSetRotation(B, R);
            dGeomSetPosition(B, x, y, z);
            both('A', q, A, B, maxc);
            dRFromAxisAndAngle(R, cos(q * 0.8), 0.6, 1, 0.5 * q + 0.03 * s);
            dGeomSetRotation(O, R);
            dGeomSetPosition(O, z, y + 0.05, x);
            both('O', q, A, O, maxc);
        }
    }
    return 0;
}

// ---- scenario 6: the other joints. One world with a hinge2 wheel, a
// universal, PR (reversed onto the world), PU, piston, plane2d, lmotor,
// dhinge, two transmissions and a null joint; limits, motors and bounce;
// bodies and joint readouts each step.
static dBodyID jb[16];
static int njb;
static dBodyID jbody(dWorldID w, double x, double y, double z, double m, double lx, double ly, double lz) {
    dBodyID b = dBodyCreate(w);
    dMass ms;
    dMassSetBoxTotal(&ms, m, lx, ly, lz);
    dBodySetMass(b, &ms);
    dBodySetPosition(b, x, y, z);
    jb[njb++] = b;
    return b;
}
static int joints_scenario(int steps) {
    dInitODE2(0);
    dRandSetSeed(4242);
    dWorldID w = dWorldCreate();
    dWorldSetGravity(w, 0, -9.81, 0);
    dWorldSetERP(w, 0.2);
    dWorldSetCFM(w, 1e-6);
    dWorldSetQuickStepNumIterations(w, 30);
    dMatrix3 R;
    dBodyID A = jbody(w, 0, 2, 0, 5, 2, 0.5, 1);
    dRFromAxisAndAngle(R, 0.2, 1, 0.1, 0.3);
    dBodySetRotation(A, R);
    dJointID ball = dJointCreateBall(w, 0);
    dJointAttach(ball, A, 0);
    dJointSetBallAnchor(ball, -0.8, 2.3, 0.1);
    dBodyID W = jbody(w, 1.1, 1.8, 0.6, 1, 0.4, 0.4, 0.2);
    dJointID h2 = dJointCreateHinge2(w, 0);
    dJointAttach(h2, A, W);
    dJointSetHinge2Anchor(h2, 1.1, 1.8, 0.6);
    dJointSetHinge2Axis1(h2, 0, 1, 0.1);
    dJointSetHinge2Axis2(h2, 0, 0, 1);
    dJointSetHinge2Param(h2, dParamSuspensionERP, 0.4);
    dJointSetHinge2Param(h2, dParamSuspensionCFM, 0.02);
    dJointSetHinge2Param(h2, dParamLoStop, -0.3);
    dJointSetHinge2Param(h2, dParamHiStop, 0.3);
    dJointSetHinge2Param(h2, dParamVel, 0.5);
    dJointSetHinge2Param(h2, dParamFMax, 1);
    dJointSetHinge2Param(h2, dParamVel2, 5);
    dJointSetHinge2Param(h2, dParamFMax2, 2);
    dBodyID U = jbody(w, -1.2, 1.5, -0.4, 1.5, 0.3, 0.8, 0.3);
    dJointID uj = dJointCreateUniversal(w, 0);
    dJointAttach(uj, A, U);
    dJointSetUniversalAnchor(uj, -1.1, 1.9, -0.3);
    dJointSetUniversalAxis1(uj, 1, 0, 0.2);
    dJointSetUniversalAxis2(uj, 0, 0.1, 1);
    dJointSetUniversalParam(uj, dParamLoStop, -0.5);
    dJointSetUniversalParam(uj, dParamHiStop, 0.5);
    dJointSetUniversalParam(uj, dParamBounce, 0.3);
    dJointSetUniversalParam(uj, dParamVel2, -1);
    dJointSetUniversalParam(uj, dParamFMax2, 0.5);
    dBodyID P = jbody(w, 3, 1, 0, 2, 0.5, 0.5, 0.5);
    dRFromAxisAndAngle(R, 1, 0, 0.3, 0.4);
    dBodySetRotation(P, R);
    dJointID pr = dJointCreatePR(w, 0);
    dJointAttach(pr, 0, P);
    dJointSetPRAnchor(pr, 3, 1.2, 0);
    dJointSetPRAxis1(pr, 0, 1, 0);
    dJointSetPRAxis2(pr, 1, 0, 0.1);
    dJointSetPRParam(pr, dParamLoStop, -0.2);
    dJointSetPRParam(pr, dParamHiStop, 0.3);
    dJointSetPRParam(pr, dParamBounce, 0.4);
    dJointSetPRParam(pr, dParamVel2, 2);
    dJointSetPRParam(pr, dParamFMax2, 1);
    dBodyID Q = jbody(w, -1.3, 0.8, -0.5, 1, 0.3, 0.3, 0.6);
    dJointID pu = dJointCreatePU(w, 0);
    dJointAttach(pu, U, Q);
    dJointSetPUAnchor(pu, -1.25, 1.1, -0.45);
    dJointSetPUAxis1(pu, 1, 0, 0);
    dJointSetPUAxis2(pu, 0, 0.2, 1);
    dJointSetPUAxis3(pu, 0, 1, 0.1);
    dJointSetPUParam(pu, dParamLoStop3, -0.1);
    dJointSetPUParam(pu, dParamHiStop3, 0.1);
    dJointSetPUParam(pu, dParamBounce3, 0.2);
    dJointSetPUParam(pu, dParamLoStop, -0.4);
    dJointSetPUParam(pu, dParamHiStop, 0.4);
    dJointSetPUParam(pu, dParamVel2, 0.7);
    dJointSetPUParam(pu, dParamFMax2, 0.3);
    dBodyID S = jbody(w, -3, 1.5, 0.5, 1.2, 0.4, 0.4, 0.4);
    dJointID ps = dJointCreatePiston(w, 0);
    dJointAttach(ps, S, 0);
    dJointSetPistonAnchor(ps, -3, 1.5, 0.5);
    dJointSetPistonAxis(ps, 0.1, 1, 0);
    dJointSetPistonParam(ps, dParamLoStop, -0.5);
    dJointSetPistonParam(ps, dParamHiStop, 0.5);
    dJointSetPistonParam(ps, dParamVel2, 3);
    dJointSetPistonParam(ps, dParamFMax2, 1);
    dBodyID T = jbody(w, 0, 0.5, 4, 1, 0.5, 0.5, 0.5);
    dJointID p2 = dJointCreatePlane2D(w, 0);
    dJointAttach(p2, T, 0);
    dJointSetPlane2DXParam(p2, dParamVel, 1);
    dJointSetPlane2DXParam(p2, dParamFMax, 5);
    dJointSetPlane2DAngleParam(p2, dParamVel, 2);
    dJointSetPlane2DAngleParam(p2, dParamFMax, 1);
    dBodyID L = jbody(w, 4, 3, -2, 2, 0.3, 0.3, 0.3);
    dRFromAxisAndAngle(R, 0, 0, 1, 0.5);
    dBodySetRotation(L, R);
    dJointID lm = dJointCreateLMotor(w, 0);
    dJointAttach(lm, L, 0);
    dJointSetLMotorNumAxes(lm, 2);
    dJointSetLMotorAxis(lm, 0, 0, 1, 0, 0);
    dJointSetLMotorAxis(lm, 1, 1, 0, 1, 0);
    dJointSetLMotorParam(lm, dParamVel, 0.5);
    dJointSetLMotorParam(lm, dParamFMax, 3);
    dJointSetLMotorParam(lm, dParamVel2, 0);
    dJointSetLMotorParam(lm, dParamFMax2, 30);
    dBodyID D1 = jbody(w, -4, 4, -3, 1, 0.3, 0.3, 0.3);
    dBodyID D2 = jbody(w, -4.5, 3.2, -3, 1, 0.3, 0.3, 0.3);
    dBodySetLinearVel(D2, 0.5, 0, 0.3);
    dJointID dh = dJointCreateDHinge(w, 0);
    dJointAttach(dh, D1, D2);
    dJointSetDHingeAxis(dh, 0, 0.3, 1);
    dJointSetDHingeAnchor1(dh, -4, 4, -3);
    dJointSetDHingeAnchor2(dh, -4.5, 3.2, -3);
    dJointSetDHingeParam(dh, dParamERP, 0.3);
    dJointID dhw = dJointCreateBall(w, 0);
    dJointAttach(dhw, D1, 0);
    dJointSetBallAnchor(dhw, -4, 4.2, -3);
    dBodyID G[3];
    double gx[3] = {2, 2.9, 5};
    for (int i = 0; i < 3; i++) {
        G[i] = jbody(w, gx[i], 5, 3, 1 + i, 0.6, 0.6, 0.1);
        dJointID h = dJointCreateHinge(w, 0);
        dJointAttach(h, G[i], 0);
        dJointSetHingeAnchor(h, gx[i], 5, 3);
        dJointSetHingeAxis(h, 0, 0, 1);
        if (i == 0) {
            dJointSetHingeParam(h, dParamVel, 3);
            dJointSetHingeParam(h, dParamFMax, 10);
        }
    }
    dJointID tr1 = dJointCreateTransmission(w, 0);
    dJointAttach(tr1, G[0], G[1]);
    dJointSetTransmissionMode(tr1, dTransmissionParallelAxes);
    dJointSetTransmissionRatio(tr1, 2);
    dJointSetTransmissionAnchor1(tr1, 2, 5, 3);
    dJointSetTransmissionAnchor2(tr1, 2.9, 5, 3);
    dJointSetTransmissionAxis(tr1, 0, 0, 1);
    dJointID tr2 = dJointCreateTransmission(w, 0);
    dJointAttach(tr2, G[0], G[2]);
    dJointSetTransmissionMode(tr2, dTransmissionChainDrive);
    dJointSetTransmissionRadius1(tr2, 0.3);
    dJointSetTransmissionRadius2(tr2, 0.5);
    dJointSetTransmissionBacklash(tr2, 0.02);
    dJointSetTransmissionAnchor1(tr2, 2, 5, 3);
    dJointSetTransmissionAnchor2(tr2, 5, 5, 3);
    dJointSetTransmissionAxis(tr2, 0, 0, 1);
    dJointID nj = dJointCreateNull(w, 0);
    dJointAttach(nj, G[1], G[2]);
    for (int s = 0; s < steps; s++) {
        dWorldQuickStep(w, 0.005);
        printf("S%d\n", s);
        for (int i = 0; i < njb; i++) dump_body(jb[i]);
        dReal a1, a2;
        printf("J");
        hex(dJointGetHinge2Angle1(h2));
        hex(dJointGetHinge2Angle2(h2));
        dJointGetUniversalAngles(uj, &a1, &a2);
        hex(a1);
        hex(a2);
        hex(dJointGetPRPosition(pr));
        hex(dJointGetPRAngle(pr));
        hex(dJointGetPUPosition(pu));
        dJointGetPUAngles(pu, &a1, &a2);
        hex(a1);
        hex(a2);
        hex(dJointGetPistonPosition(ps));
        hex(dJointGetPistonAngle(ps));
        hex(dJointGetTransmissionAngle1(tr1));
        hex(dJointGetTransmissionAngle2(tr2));
        dVector3 cp;
        dJointGetTransmissionContactPoint1(tr2, cp);
        for (int k = 0; k < 3; k++) hex(cp[k]);
        printf("\n");
    }
    return 0;
}
