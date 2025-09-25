! FORTRAN 77 (Yeah Baby!) code used to "re-navigate" the altitude of a GLM event from the standard lightning ellipse altitude to
! another altitude. This can be used, for example, to adjust the height of a bolide event from lightning altitudes 
! (typically 10 km) to bolide heights (typically 80 km).

!****************************************************************************
      subroutine renavigate(lat1, lon1, sat_lon, e1, p1, e2, p2,
     *                      lat2,lon2,x2,y2,z2)
!
!     Re-navigates GLM event geodetic coordinates from the original lightning
!     ellipsoid defined by e1 and p1 to the new lightning ellispoid defined by
!     e2 and p2.  This re-navigation calculation uses a ray-ellipsoid
!     intersection calculation in ECEF (Earth-Centered, Earth-Fixed)
!     coordinates, similar to the calculation made in the original navigation
!     of GLM events. As of September 2018, e1 = 14 km and p1 = 6 km.
!     this code was given to NASA by Lockheed Martin (Clem Tillier)
!
! Inputs
!   lat1 -- Original navigated event latitude (deg).  Specified as a vector
!           of the same dimensions as lon1.
!   lon1 -- Original navigated event longitude (deg).  Specified as a
!           vector of arbitrary length.
!   sat_lon -- [float] longitude of satellite (latitude assumed to be 0.0 degrees)
!   e1   -- Lightning ellipsoid altitude, equatorial, to which lat1 an lon1
!           were originally navigated. Specified as km altitude above Earth
!           ellipsoid equatorial radius. Original GLM value was 16 km.
!   p1   -- Lightning ellipsoid altitude, polar, to which lat1 and lon1 were
!           originally navigated. Specified as km altitude above Earth
!           ellipsoid polar radius. Original GLM value was 6 km.
!   e2   -- New lightning ellipsoid altitude, equatorial.
!   p2   -- New lightning ellipsoid altitude, polar
!
! Outputs
!   lat2 -- Re-navigated event latitude (deg)
!   lon2 -- Re-navigated event longitude (deg)
!   x2   -- New event coordinates (in ECEF)
!   y2   -- New event coordinates (in ECEF)
!   z2   -- New event coordinates (in ECEF)
!
! Note, if an event cannot be navigated to the new ellispoid (e.g. because
! the view direction from the satellite doesn't intersect with the new '
! ellipsoid), the corresponding lon2/lat2 values are NaN.
!
! Testing shows that round-trip residual errors (re-navigating to another
! ellipsoid then back to the original ellipsoid) are on the order of 1e-12
! degrees.
!
!     satellite longitude (deg) GOES positions: East -75; Test -89.5; West -137
!     satellite orbit radius (km)
!     Earth equatorial radius (km)
!     Earth flattening factor (a-b)/a
!
!
      implicit double precision (o-z,a-h)
!     Set inputs/outputs for Python interface
      double precision lon1, lat1, lon2, lat2
!f2py intent(in) lon1, lat1, sat_lon, e1, p1, e2, p2,
!f2py intent(out) lon2, lat2, x2, y2, z2
!     Some constants      
      parameter (sat_radius = 42164.0d0) 
      parameter (eer = 6378.137d0)
      parameter (eff = 3.35281d-3)
      parameter (pi = 4.0d0 * datan(1.0d0))
      dtr = pi/180.0d0
      rtd = 180.0d0/pi
!
!     Step 1: Sanity check the inputs
!
      if (sat_lon.lt.-180.0.or.sat_lon.gt.180.0) then
        write(*,*) 'Warning: satellite longitude outside valid range'
        stop
      end if
!
      if (e1 .lt. p1) then
        write(*,*) 'Warning polar 1 altitude greater than equatorial
     * 1 altitude'
        stop
      end if
!
      if (e2 .lt. p2) then
        write(*,*) 'Warning polar 2 altitude greater than equatorial 
     * 2 altitude'
        stop
      end if
!
!     Step 2: convert input longitude and latitude to ECEF (Earth centered,
!     Earth fixed) vectors.
!     ECEF x passes through prime meridian at equator (lat = lon = 0).
!     ECEF y passes through equator at 90 deg east.
!     ECEF z passes through the north pole.
!     calculate original lightning ellipsoid equatorial and polar radii
!     compute the original lightning ellipsoid's flattening factor '
!     convert to geocentric latitude from geodetic latitude
!
      re1 = eer + e1
      rp1 = (1.0d0-eff)*eer + p1
      ff1 = (re1-rp1)/re1
      rlat1prime = datan((1.0d0-ff1)**2*dtan(dtr*lat1))
!
!     compute the x,y,z ECEF coordinates on the original lightning ellipsoid
!
      rmag = re1*(1.0d0-ff1)/dsqrt(1.0d0-ff1*(2.0d0-ff1)*
     *       dcos(rlat1prime)**2)
!
!     ellipsoid radius
!
      x1 = rmag*dcos(rlat1prime)*dcos(dtr*lon1)
      y1 = rmag*dcos(rlat1prime)*dsin(dtr*lon1)
      z1 = rmag*dsin(rlat1prime)
!
!     Step 3: compute pointing vector d from satellite (independent of
!     ellipsoid).  This is simply the vector from the center of the Earth to
!     the lightning event, minus the vector from the center of the Earth to the
!     satellite-- and yields the vector from the satellite to the lightning
!     event.  This line of sight vector will later be intersected with the new
!     ellipsoid.
!
      Rx = sat_radius * dcos(dtr*sat_lon)
      Ry = sat_radius * dsin(dtr*sat_lon)
      Rz = 0.0d0
      dx = x1 - Rx
      dy = y1 - Ry
      dz = z1 - Rz
      unorm = dsqrt(dx*dx + dy*dy + dz*dz)
      dx = dx/unorm
      dy = dy/unorm
      dz = dz/unorm
!
!     Step 4: compute the intersection of a ray parallel to (dx,dy,dz) and the
!     new ellipsoid.  This calculation is performed according to CDRL046
!     navigation design document, appendix J, using the quadratic formulation
!     of equations J-6 through J-20.
!     new ellipsoid parameters
!
      re2 = eer + e2
      rp2 = (1.0d0-eff)*eer + p2
      ff2 = (re2-rp2)/re2
!
!     CDRL046 Equation J-13
!
      Q1 = dx*dx + dy*dy + dz*dz/((1.0d0-ff2)**2)
      Q2 = 2.0d0*(Rx*dx + Ry*dy) 
!
!     Rz is zero, neglect + Rz*dz./((1-Fle)^2));
!
      Q3 = Rx*Rx + Ry*Ry - re2*re2 
!
!     Rz is zero, neglect + Rz*Rz./((1-Fle)^2)
!
      Q4 = Q2*Q2 - 4.0d0*Q1*Q3 
!
!     quadratic equation determinant (Equation J-15)
!     any members of Q4 that are negative cannot be navigated to (do not
!     intersect with) the new ellipsoid.  Set those to NaN, which will properly
!     propagate to the output where unnavigable lat2/lon2 values will be NaN.
!
      if (Q4 .lt. 0.0d0) then
        write(*,*) 'unnavigable'
      end if
!
      D = (-Q2 - dsqrt(Q4))/(2.0d0*Q1)
!
!     quadratic root (Equation J-16)
!
      x2 = Rx + D*dx 
!
!     Equation J-8.  (x2,y2,z2) is the intersection of the 
!
      y2 = Ry + D*dy 
!
!     line of sight ray with the new lightning ellipsoid.
!
      z2 = Rz + D*dz
!
!     Step 5: convert the ECEF coordinates of the new lightning ellipsoid
!     intersection back into geodetic coordinates.
!
      rnormp = dsqrt(x2*x2 + y2*y2 + z2*z2)
!
!     new longitude in degrees, Eqn J-19
!
      lon2 = datan2(y2/rnormp, x2/rnormp)*rtd
!
!     geocentric latitude (not what we want, Eqn J-18)
!
      phi_prime = dasin(z2/rnormp)
!
!     Eqn J-19
!
      lat2 = datan(1.0d0/(1.0d0-ff1)**2*dtan(phi_prime))*rtd
!
      return
      end

!*************************************************************************************************************
! FORTRAN 77 routine to compute the altitude of an event based on the X,Y,Z coordinates returned by renavigate
!
! No documentation was provide with this code. Your guess to functionality or correctness is as good as mine!
!
      Subroutine event_altitude (X, Y, Z, ALAT, ALON, ALT)
!
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
!     Set inputs/outputs for Python interface
!f2py intent(in) X, Y, Z
!f2py intent(out) ALAT, ALON, ALT
      PARAMETER ( DTA = 1.0D-5, ALRG = 1.0D30 )
      PARAMETER (RE = 6378.137D0)
      PARAMETER (eCC = 0.0818191908426D0)
!
      XX = DSQRT(Y*Y +  X*X)
      YY = Z
      CK3 = 1.0d0 / (1.0d0 - ecc*ecc)
      TANPHI = YY * CK3 / XX
      TANPHO = ALRG
!
      IF (TANPHI .NE. 0.0D0) THEN
!
 100    CONTINUE
!
!        RESID = DABS (( TANPHO – TANPHI ) / TANPHI )
        resid = dabs((tanpho - tanphi) / tanphi)
!
        IF ( RESID .GT. DTA ) THEN
!
          TANPHO = TANPHI
          DENO = DSQRT ( 1.0D0 + ( TANPHI*TANPHI / CK3 ))
!          TANPHI = YY / ( XX – RE * ECC * ECC / DENO )
          tanphi = yy / (xx - re *ecc*ecc / deno)
!
          GOTO 100
!
        END IF
!
      END IF
!
      ALAT = DATAN2 ( TANPHI, 1.0D0 )
      ALON = DATAN2 ( Y, X )
      COS_SQ_PHI = 1.0D0 / ( 1.0D0 + TANPHI * TANPHI )
      COS_PHI = DSQRT ( COS_SQ_PHI )
      SIN_SQ_PHI = 1.0D0 - COS_SQ_PHI
      SIN_PHI = DSQRT ( SIN_SQ_PHI )
      ABS_Z = DABS ( Z )
!
      IF ( ABS_Z .LT. XX ) THEN
!
        ROC = RE / DSQRT ( 1.0D0 - ECC*ECC*SIN_SQ_PHI )
        ALT = XX / COS_PHI - ROC
!
      ELSE
!
        ALT = ABS_Z / SIN_PHI - RE * ( 1.0D0 - ECC*ECC) / 
     *        DSQRT ( 1.0D0 - ECC*ECC*SIN_SQ_PHI)
!
      END IF
!
      RETURN
      END

